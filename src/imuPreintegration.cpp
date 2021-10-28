#include "utility.h"
#include "param_server.h"

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>

using gtsam::symbol_shorthand::P; // Pose3 (x,y,z,r,p,y)
using gtsam::symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using gtsam::symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)

gtsam::Pose3 makeGtsamPose(const geometry_msgs::Pose & pose)
{
  const geometry_msgs::Point p = pose.position;
  const geometry_msgs::Quaternion q = pose.orientation;
  return gtsam::Pose3(
    gtsam::Rot3::Quaternion(q.w, q.x, q.y, q.z),
    gtsam::Point3(p.x, p.y, p.z)
  );
}

boost::shared_ptr<gtsam::PreintegrationParams> initialIntegrationParams(
  const float imuGravity, const float imuAccNoise, const float imuGyrNoise)
{
  boost::shared_ptr<gtsam::PreintegrationParams> p =
    gtsam::PreintegrationParams::MakeSharedU(imuGravity);
  // acc white noise in continuous
  p->accelerometerCovariance = gtsam::Matrix33::Identity(3, 3) * pow(imuAccNoise, 2);
  // gyro white noise in continuous
  p->gyroscopeCovariance = gtsam::Matrix33::Identity(3, 3) * pow(imuGyrNoise, 2);
  // error committed in integrating position from velocities
  p->integrationCovariance = gtsam::Matrix33::Identity(3, 3) * pow(1e-4, 2);
  return p;
}

tf::Transform getLidarToBaseLink(
  const std::string & lidarFrame,
  const std::string & baselinkFrame)
{
  if (lidarFrame == baselinkFrame) {
    return identityTransform();
  }

  tf::StampedTransform transform;
  try {
    tf::TransformListener listener;
    listener.lookupTransform(lidarFrame, baselinkFrame, ros::Time(0), transform);
  } catch (tf::TransformException ex) {
    ROS_ERROR("%s", ex.what());
  }

  return transform;
}

void popOldMessages(
  const double time_threshold,
  double & last_imu_time,
  std::deque<sensor_msgs::Imu> & imu_queue)
{
  while (!imu_queue.empty() && timeInSec(imu_queue.front().header) < time_threshold) {
    last_imu_time = timeInSec(imu_queue.front().header);
    imu_queue.pop_front();
  }
}

class OdomToBaselink
{
private:
  const tf::Transform lidar_to_baselink;
  const std::string odometryFrame;
  const std::string baselinkFrame;

public:
  OdomToBaselink(
    const std::string & lidarFrame,
    const std::string & odometryFrame,
    const std::string & baselinkFrame)
  : lidar_to_baselink(getLidarToBaseLink(lidarFrame, baselinkFrame)),
    odometryFrame(odometryFrame),
    baselinkFrame(baselinkFrame)
  {
  }

  tf::StampedTransform get(
    const geometry_msgs::Pose & odometry,
    const ros::Time & timestamp) const
  {
    const tf::Transform lidar_odometry = poseMsgToTF(odometry);
    return tf::StampedTransform(
      lidar_odometry * lidar_to_baselink,
      timestamp, odometryFrame, baselinkFrame);
  }
};

class ImuPath
{
private:
  nav_msgs::Path imuPath;

  const std::string odometryFrame;

public:
  ImuPath(const std::string odometryFrame)
  : odometryFrame(odometryFrame) {}

  nav_msgs::Path make(
    const ros::Time & timestamp,
    const geometry_msgs::Pose & pose,
    const double lidarOdomTime)
  {
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header.stamp = timestamp;
    pose_stamped.header.frame_id = odometryFrame;
    pose_stamped.pose = pose;
    imuPath.poses.push_back(pose_stamped);
    while (
      !imuPath.poses.empty() &&
      imuPath.poses.front().header.stamp.toSec() < lidarOdomTime - 1.0)
    {
      imuPath.poses.erase(imuPath.poses.begin());
    }
    imuPath.header.stamp = timestamp;
    imuPath.header.frame_id = odometryFrame;

    return imuPath;
  }
};

Eigen::Affine3d latestOdometry(
  const geometry_msgs::Pose & front_pose,
  const geometry_msgs::Pose & back_pose,
  const geometry_msgs::Pose & lidar_odom)
{
  const Eigen::Affine3d front = poseToAffine(front_pose);
  const Eigen::Affine3d back = poseToAffine(back_pose);
  const Eigen::Affine3d incre = front.inverse() * back;
  return poseToAffine(lidar_odom) * incre;
}

class TransformFusion : public ParamServer
{
public:
  std::mutex mtx;

  const ros::Subscriber subLaserOdometry;
  const ros::Subscriber subImuOdometry;

  const ros::Publisher pubImuOdometry;
  const ros::Publisher pubImuPath;

  geometry_msgs::Pose lidar_odom;

  const tf::Transform lidar_to_baselink;
  const OdomToBaselink odom_to_baselink;
  tf::TransformBroadcaster broadcaster;

  double lidarOdomTime = -1;
  std::deque<nav_msgs::Odometry> imuOdomQueue;

  tf::TransformBroadcaster tfMap2Odom;

  ImuPath imu_path;

  TransformFusion()
  : subLaserOdometry(nh.subscribe<nav_msgs::Odometry>(
        "lio_sam/mapping/odometry",
        5, &TransformFusion::lidarOdometryHandler, this,
        ros::TransportHints().tcpNoDelay())),
    subImuOdometry(nh.subscribe<nav_msgs::Odometry>(
        odomTopic + "_incremental",
        2000, &TransformFusion::imuOdometryHandler, this,
        ros::TransportHints().tcpNoDelay())),
    pubImuOdometry(nh.advertise<nav_msgs::Odometry>(odomTopic, 2000)),
    pubImuPath(nh.advertise<nav_msgs::Path>("lio_sam/imu/path", 1)),
    odom_to_baselink(OdomToBaselink(lidarFrame, odometryFrame, baselinkFrame)),
    imu_path(ImuPath(odometryFrame))
  {
  }

  void lidarOdometryHandler(const nav_msgs::Odometry::ConstPtr & odom_msg)
  {
    std::lock_guard<std::mutex> lock(mtx);

    lidar_odom = odom_msg->pose.pose;

    lidarOdomTime = odom_msg->header.stamp.toSec();
  }

  void imuOdometryHandler(const nav_msgs::Odometry::ConstPtr & odom_msg)
  {
    tfMap2Odom.sendTransform(
      tf::StampedTransform(identityTransform(), odom_msg->header.stamp, mapFrame, odometryFrame));

    std::lock_guard<std::mutex> lock(mtx);

    imuOdomQueue.push_back(*odom_msg);

    // get latest odometry (at current IMU stamp)
    if (lidarOdomTime == -1) {
      return;
    }

    dropBefore(lidarOdomTime, imuOdomQueue);
    const Eigen::Affine3d last =
      latestOdometry(imuOdomQueue.front().pose.pose, imuOdomQueue.back().pose.pose, lidar_odom);

    // publish latest odometry
    nav_msgs::Odometry laserOdometry = imuOdomQueue.back();
    laserOdometry.pose.pose = affineToPose(last);
    pubImuOdometry.publish(laserOdometry);

    broadcaster.sendTransform(
      odom_to_baselink.get(laserOdometry.pose.pose, odom_msg->header.stamp));

    if (pubImuPath.getNumSubscribers() != 0) {
      pubImuPath.publish(
        imu_path.make(imuOdomQueue.back().header.stamp, laserOdometry.pose.pose, lidarOdomTime));
    }
  }
};

bool failureDetection(
  const gtsam::Vector3 & velocity,
  const gtsam::imuBias::ConstantBias & bias)
{
  if (velocity.norm() > 30) {
    ROS_WARN("Large velocity, reset IMU-preintegration!");
    return true;
  }

  const Eigen::Vector3d ba = bias.accelerometer();
  const Eigen::Vector3d bg = bias.gyroscope();
  if (ba.norm() > 1.0 || bg.norm() > 1.0) {
    ROS_WARN("Large bias, reset IMU-preintegration!");
    return true;
  }

  return false;
}

using Diagonal = gtsam::noiseModel::Diagonal;

void resetOptimizer(
  const gtsam::Pose3 & pose,
  const gtsam::Vector3 & velocity,
  const gtsam::imuBias::ConstantBias & bias,
  const int key,
  gtsam::ISAM2 & optimizer)
{
  const auto pose_noise =
    gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(P(key - 1)));
  const auto velocity_noise =
    gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(V(key - 1)));
  const auto bias_noise =
    gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(B(key - 1)));

  const gtsam::ISAM2Params params(gtsam::ISAM2GaussNewtonParams(), 0.1, 1);
  optimizer = gtsam::ISAM2(params);

  gtsam::NonlinearFactorGraph graph;

  graph.add(gtsam::PriorFactor<gtsam::Pose3>(P(0), pose, pose_noise));
  graph.add(gtsam::PriorFactor<gtsam::Vector3>(V(0), velocity, velocity_noise));
  graph.add(gtsam::PriorFactor<gtsam::imuBias::ConstantBias>(B(0), bias, bias_noise));

  gtsam::Values values;
  values.insert(P(0), pose);
  values.insert(V(0), velocity);
  values.insert(B(0), bias);

  optimizer.update(graph, values);
}

gtsam::ISAM2 initOptimizer(const gtsam::Pose3 & lidar_to_imu, const gtsam::Pose3 & lidar_pose)
{
  const gtsam::ISAM2Params params(gtsam::ISAM2GaussNewtonParams(), 0.1, 1);

  gtsam::NonlinearFactorGraph graph;

  const Diagonal::shared_ptr pose_noise(Diagonal::Sigmas(1e-2 * Vector6d::Ones()));
  // rad,rad,rad, m, m, m (m/s)
  const Diagonal::shared_ptr velocity_noise(gtsam::noiseModel::Isotropic::Sigma(3, 1e4));
  // 1e-2 ~ 1e-3 seems to be good
  const Diagonal::shared_ptr bias_noise(gtsam::noiseModel::Isotropic::Sigma(6, 1e-3));

  gtsam::Pose3 pose = lidar_pose.compose(lidar_to_imu);
  graph.add(gtsam::PriorFactor<gtsam::Pose3>(P(0), pose, pose_noise));

  gtsam::Vector3 velocity = gtsam::Vector3(0, 0, 0);
  graph.add(gtsam::PriorFactor<gtsam::Vector3>(V(0), velocity, velocity_noise));

  gtsam::imuBias::ConstantBias bias = gtsam::imuBias::ConstantBias();
  graph.add(gtsam::PriorFactor<gtsam::imuBias::ConstantBias>(B(0), bias, bias_noise));

  gtsam::Values values;
  values.insert(P(0), pose);
  values.insert(V(0), velocity);
  values.insert(B(0), bias);

  gtsam::ISAM2 optimizer = gtsam::ISAM2(params);
  optimizer.update(graph, values);
  return optimizer;
}

void imuPreIntegration(
  const double time_threshold,
  const gtsam::imuBias::ConstantBias & prev_odom_bias_,
  gtsam::PreintegratedImuMeasurements & imuIntegratorImu_,
  std::deque<sensor_msgs::Imu> & imu_queue)
{
  // first pop imu message older than current correction data
  double last_imu_time = -1;
  popOldMessages(time_threshold, last_imu_time, imu_queue);

  // repropogate
  if (imu_queue.empty()) {
    return;
  }

  // reset bias use the newly optimized bias
  imuIntegratorImu_.resetIntegrationAndSetBias(prev_odom_bias_);
  // integrate imu message from the beginning of this optimization
  for (unsigned int i = 0; i < imu_queue.size(); ++i) {
    const sensor_msgs::Imu & msg = imu_queue[i];
    const double imu_time = timeInSec(msg.header);
    const double dt = (last_imu_time < 0) ? (1.0 / 500.0) : (imu_time - last_imu_time);
    imuIntegratorImu_.integrateMeasurement(
      vector3ToEigen(msg.linear_acceleration),
      vector3ToEigen(msg.angular_velocity),
      dt
    );
    last_imu_time = imu_time;
  }
}

void imuIntegration(
  const double time_threshold, double & last_imu_time_opt,
  gtsam::PreintegratedImuMeasurements & imuIntegratorOpt_,
  std::deque<sensor_msgs::Imu> & imuQueOpt)
{
  while (!imuQueOpt.empty() && timeInSec(imuQueOpt.front().header) < time_threshold) {
    // pop and integrate imu data that is between two optimizations
    const sensor_msgs::Imu & front = imuQueOpt.front();
    const double imu_time = timeInSec(front.header);
    const double dt = (last_imu_time_opt < 0) ? (1.0 / 500.0) : (imu_time - last_imu_time_opt);

    imuIntegratorOpt_.integrateMeasurement(
      vector3ToEigen(front.linear_acceleration),
      vector3ToEigen(front.angular_velocity),
      dt
    );

    last_imu_time_opt = imu_time;
    imuQueOpt.pop_front();
  }
}

class IMUPreintegration : public ParamServer
{
public:
  std::mutex mtx;

  const ros::Subscriber subImu;
  const ros::Subscriber subOdometry;
  const ros::Publisher pubImuOdometry;

  const boost::shared_ptr<gtsam::PreintegrationParams> integration_params_;
  const gtsam::imuBias::ConstantBias prior_imu_bias_;

  const gtsam::Vector between_noise_bias_;

  gtsam::PreintegratedImuMeasurements imuIntegratorImu_;
  gtsam::PreintegratedImuMeasurements imuIntegratorOpt_;

  bool systemInitialized;

  std::deque<sensor_msgs::Imu> imuQueOpt;
  std::deque<sensor_msgs::Imu> imu_queue;

  gtsam::Pose3 prev_pose_;
  gtsam::Vector3 prev_velocity_;
  gtsam::NavState prev_state_;
  gtsam::imuBias::ConstantBias prev_bias_;

  bool doneFirstOpt = false;
  double last_imu_time = -1;
  double last_imu_time_opt = -1;

  gtsam::ISAM2 optimizer;

  const double delta_t = 0;

  int key = 1;

  gtsam::Pose3 imu2Lidar = gtsam::Pose3(
    gtsam::Rot3(1, 0, 0, 0),
    gtsam::Point3(-extTrans.x(), -extTrans.y(), -extTrans.z()));
  gtsam::Pose3 lidar_to_imu = gtsam::Pose3(
    gtsam::Rot3(1, 0, 0, 0),
    gtsam::Point3(extTrans.x(), extTrans.y(), extTrans.z()));

  IMUConverter imu_converter_;

  IMUPreintegration()
  : subImu(nh.subscribe<sensor_msgs::Imu>(
        imuTopic, 2000, &IMUPreintegration::imuHandler,
        this, ros::TransportHints().tcpNoDelay())),
    subOdometry(nh.subscribe<nav_msgs::Odometry>(
        "lio_sam/mapping/odometry_incremental", 5, &IMUPreintegration::odometryHandler,
        this, ros::TransportHints().tcpNoDelay())),
    pubImuOdometry(nh.advertise<nav_msgs::Odometry>(odomTopic + "_incremental", 2000)),
    integration_params_(initialIntegrationParams(imuGravity, imuAccNoise, imuGyrNoise)),
    prior_imu_bias_(Eigen::Matrix<double, 1, 6>::Zero()),
    between_noise_bias_(
      (Vector6d() <<
        imuAccBiasN, imuAccBiasN, imuAccBiasN,
        imuGyrBiasN, imuGyrBiasN, imuGyrBiasN).finished()),
    imuIntegratorImu_(gtsam::PreintegratedImuMeasurements(integration_params_, prior_imu_bias_)),
    imuIntegratorOpt_(gtsam::PreintegratedImuMeasurements(integration_params_, prior_imu_bias_)),
    systemInitialized(false)
  {
  }

  void odometryHandler(const nav_msgs::Odometry::ConstPtr & odom_msg)
  {
    std::lock_guard<std::mutex> lock(mtx);

    const double odom_time = timeInSec(odom_msg->header);

    // make sure we have imu data to integrate
    if (imuQueOpt.empty()) {
      return;
    }

    const gtsam::Pose3 lidar_pose = makeGtsamPose(odom_msg->pose.pose);

    // 0. initialize system
    if (!systemInitialized) {
      // pop old IMU message
      popOldMessages(odom_time - delta_t, last_imu_time_opt, imuQueOpt);

      optimizer = initOptimizer(lidar_to_imu, lidar_pose);

      imuIntegratorImu_.resetIntegrationAndSetBias(prev_bias_);
      imuIntegratorOpt_.resetIntegrationAndSetBias(prev_bias_);

      key = 1;
      systemInitialized = true;
      return;
    }

    // reset graph for speed
    if (key == 100) {
      resetOptimizer(prev_pose_, prev_velocity_, prev_bias_, key, optimizer);

      key = 1;
    }

    // 1. integrate imu data and optimize
    imuIntegration(odom_time - delta_t, last_imu_time_opt, imuIntegratorOpt_, imuQueOpt);

    gtsam::NonlinearFactorGraph graph;

    graph.add(
      gtsam::ImuFactor(P(key - 1), V(key - 1), P(key), V(key), B(key - 1), imuIntegratorOpt_));

    graph.add(
      gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(
        B(key - 1), B(key), gtsam::imuBias::ConstantBias(),
        Diagonal::Sigmas(sqrt(imuIntegratorOpt_.deltaTij()) * between_noise_bias_))
    );

    const Diagonal::shared_ptr correctionNoise(
      Diagonal::Sigmas((Vector6d() << 0.05, 0.05, 0.05, 0.1, 0.1, 0.1).finished())
    );
    const Diagonal::shared_ptr correctionNoise2(Diagonal::Sigmas(Vector6d::Ones()));

    const auto noise = odom_msg->pose.covariance[0] == 1 ? correctionNoise2 : correctionNoise;
    graph.add(gtsam::PriorFactor<gtsam::Pose3>(P(key), lidar_pose.compose(lidar_to_imu), noise));
    // insert predicted values
    const gtsam::NavState state = imuIntegratorOpt_.predict(prev_state_, prev_bias_);

    gtsam::Values values;
    values.insert(P(key), state.pose());
    values.insert(V(key), state.v());
    values.insert(B(key), prev_bias_);

    // optimize
    optimizer.update(graph, values);
    optimizer.update();

    // Overwrite the beginning of the preintegration for the next step.
    const gtsam::Values result = optimizer.calculateEstimate();
    prev_pose_ = result.at<gtsam::Pose3>(P(key));
    prev_velocity_ = result.at<gtsam::Vector3>(V(key));
    prev_state_ = gtsam::NavState(prev_pose_, prev_velocity_);
    prev_bias_ = result.at<gtsam::imuBias::ConstantBias>(B(key));
    // Reset the optimization preintegration object.
    imuIntegratorOpt_.resetIntegrationAndSetBias(prev_bias_);
    // check optimization
    if (failureDetection(prev_velocity_, prev_bias_)) {
      last_imu_time = -1;
      doneFirstOpt = false;
      systemInitialized = false;
      return;
    }

    // 2. after optiization, re-propagate imu odometry preintegration
    imuPreIntegration(odom_time - delta_t, prev_bias_, imuIntegratorImu_, imu_queue);

    ++key;
    doneFirstOpt = true;
  }

  void imuHandler(const sensor_msgs::Imu::ConstPtr & imu_raw)
  {
    std::lock_guard<std::mutex> lock(mtx);

    const sensor_msgs::Imu imu = [&] {
        try {
          return imu_converter_.imuConverter(*imu_raw);
        } catch (const std::runtime_error & e) {
          ROS_ERROR(e.what());
          ros::shutdown();
          return sensor_msgs::Imu();
        }
      } ();

    imuQueOpt.push_back(imu);
    imu_queue.push_back(imu);

    if (!doneFirstOpt) {
      return;
    }

    const double imu_time = timeInSec(imu.header);
    const double dt = (last_imu_time < 0) ? (1.0 / 500.0) : (imu_time - last_imu_time);
    last_imu_time = imu_time;

    const Eigen::Vector3d linear_acceleration = vector3ToEigen(imu.linear_acceleration);
    const Eigen::Vector3d angular_velocity = vector3ToEigen(imu.angular_velocity);
    imuIntegratorImu_.integrateMeasurement(linear_acceleration, angular_velocity, dt);

    // predict odometry
    const gtsam::NavState current_imu = imuIntegratorImu_.predict(prev_state_, prev_bias_);

    // publish odometry
    nav_msgs::Odometry odometry;
    odometry.header.stamp = imu.header.stamp;
    odometry.header.frame_id = odometryFrame;
    odometry.child_frame_id = "odom_imu";

    // transform imu pose to ldiar
    const gtsam::Pose3 pose = current_imu.pose().compose(imu2Lidar);

    odometry.pose.pose = makePose(
      eigenToQuaternion(pose.rotation().toQuaternion()),
      eigenToPoint(pose.translation())
    );

    odometry.twist.twist = makeTwist(
      eigenToVector3(angular_velocity + prev_bias_.gyroscope()),
      eigenToVector3(current_imu.velocity())
    );
    pubImuOdometry.publish(odometry);
  }
};


int main(int argc, char ** argv)
{
  ros::init(argc, argv, "roboat_loam");

  IMUPreintegration ImuP;

  TransformFusion TF;

  ROS_INFO("\033[1;32m----> IMU Preintegration Started.\033[0m");

  ros::MultiThreadedSpinner spinner(4);
  spinner.spin();

  return 0;
}
