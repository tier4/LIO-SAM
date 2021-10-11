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

using gtsam::symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
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
    listener.waitForTransform(lidarFrame, baselinkFrame, ros::Time(0), ros::Duration(3.0));
    listener.lookupTransform(lidarFrame, baselinkFrame, ros::Time(0), transform);
  } catch (tf::TransformException ex) {
    ROS_ERROR("%s", ex.what());
  }

  return transform;
}

class TransformFusion : public ParamServer
{
public:
  std::mutex mtx;

  const ros::Subscriber subLaserOdometry;
  const ros::Subscriber subImuOdometry;

  const ros::Publisher pubImuOdometry;
  const ros::Publisher pubImuPath;

  Eigen::Affine3d lidarOdomAffine;

  const tf::Transform lidar_to_baselink;

  double lidarOdomTime = -1;
  std::deque<nav_msgs::Odometry> imuOdomQueue;

  tf::TransformBroadcaster tfMap2Odom;

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
    lidar_to_baselink(getLidarToBaseLink(lidarFrame, baselinkFrame))
  {
  }

  void lidarOdometryHandler(const nav_msgs::Odometry::ConstPtr & odom_msg)
  {
    std::lock_guard<std::mutex> lock(mtx);

    lidarOdomAffine = poseToAffine(odom_msg->pose.pose);

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
    const Eigen::Affine3d front = poseToAffine(imuOdomQueue.front().pose.pose);
    const Eigen::Affine3d back = poseToAffine(imuOdomQueue.back().pose.pose);
    const Eigen::Affine3d incre = front.inverse() * back;
    const Eigen::Affine3d last = lidarOdomAffine * incre;

    // publish latest odometry
    nav_msgs::Odometry laserOdometry = imuOdomQueue.back();
    laserOdometry.pose.pose = affineToPose(last);
    pubImuOdometry.publish(laserOdometry);

    // publish tf
    tf::TransformBroadcaster broadcaster;
    tf::Transform lidar_odometry;
    tf::poseMsgToTF(laserOdometry.pose.pose, lidar_odometry);
    broadcaster.sendTransform(
      tf::StampedTransform(
        lidar_odometry * lidar_to_baselink,
        odom_msg->header.stamp, odometryFrame, baselinkFrame));

    // publish IMU path
    static nav_msgs::Path imuPath;
    static double last_path_time = -1;
    const double imuTime = imuOdomQueue.back().header.stamp.toSec();
    if (imuTime - last_path_time > 0.1) {
      last_path_time = imuTime;
      geometry_msgs::PoseStamped pose_stamped;
      pose_stamped.header.stamp = imuOdomQueue.back().header.stamp;
      pose_stamped.header.frame_id = odometryFrame;
      pose_stamped.pose = laserOdometry.pose.pose;
      imuPath.poses.push_back(pose_stamped);
      while (!imuPath.poses.empty() &&
        imuPath.poses.front().header.stamp.toSec() < lidarOdomTime - 1.0)
      {
        imuPath.poses.erase(imuPath.poses.begin());
      }
      if (pubImuPath.getNumSubscribers() != 0) {
        imuPath.header.stamp = imuOdomQueue.back().header.stamp;
        imuPath.header.frame_id = odometryFrame;
        pubImuPath.publish(imuPath);
      }
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
  const gtsam::Pose3 & prevPose_,
  const gtsam::Vector3 & prevVel_,
  const gtsam::imuBias::ConstantBias & prevBias_,
  const int key,
  gtsam::ISAM2 & optimizer)
{
  // get updated noise before reset
  const auto updatedPoseNoise =
    gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(X(key - 1)));
  const auto updatedVelNoise =
    gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(V(key - 1)));
  const auto updatedBiasNoise =
    gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(B(key - 1)));

  // reset graph
  const gtsam::ISAM2Params params(gtsam::ISAM2GaussNewtonParams(), 0.1, 1);
  optimizer = gtsam::ISAM2(params);

  gtsam::NonlinearFactorGraph newGraphFactors;
  gtsam::NonlinearFactorGraph graphFactors = newGraphFactors;

  // add pose
  gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, updatedPoseNoise);
  graphFactors.add(priorPose);
  // add velocity
  gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, updatedVelNoise);
  graphFactors.add(priorVel);
  // add bias
  gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, updatedBiasNoise);
  graphFactors.add(priorBias);
  // add values
  gtsam::Values graphValues;
  graphValues.insert(X(0), prevPose_);
  graphValues.insert(V(0), prevVel_);
  graphValues.insert(B(0), prevBias_);
  // optimize once
  optimizer.update(graphFactors, graphValues);
}

gtsam::ISAM2 initOptimizer(const gtsam::Pose3 & lidar2Imu, const gtsam::Pose3 & lidar_pose)
{
  const gtsam::ISAM2Params params(gtsam::ISAM2GaussNewtonParams(), 0.1, 1);

  gtsam::NonlinearFactorGraph graphFactors;

  const Diagonal::shared_ptr priorPoseNoise(Diagonal::Sigmas(1e-2 * Vector6d::Ones()));
  // rad,rad,rad, m, m, m (m/s)
  const Diagonal::shared_ptr priorVelNoise(gtsam::noiseModel::Isotropic::Sigma(3, 1e4));
  // 1e-2 ~ 1e-3 seems to be good
  const Diagonal::shared_ptr priorBiasNoise(gtsam::noiseModel::Isotropic::Sigma(6, 1e-3));

  // initial pose
  gtsam::Pose3 prevPose_ = lidar_pose.compose(lidar2Imu);
  gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, priorPoseNoise);
  graphFactors.add(priorPose);
  // initial velocity
  gtsam::Vector3 prevVel_ = gtsam::Vector3(0, 0, 0);
  gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, priorVelNoise);
  graphFactors.add(priorVel);
  // initial bias
  gtsam::imuBias::ConstantBias prevBias_ = gtsam::imuBias::ConstantBias();
  gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, priorBiasNoise);
  graphFactors.add(priorBias);
  // add values
  gtsam::Values graphValues;
  graphValues.insert(X(0), prevPose_);
  graphValues.insert(V(0), prevVel_);
  graphValues.insert(B(0), prevBias_);
  // optimize once
  gtsam::ISAM2 optimizer = gtsam::ISAM2(params);
  optimizer.update(graphFactors, graphValues);
  return optimizer;
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

  const gtsam::Vector noiseModelBetweenBias;

  gtsam::PreintegratedImuMeasurements imuIntegratorImu_;
  gtsam::PreintegratedImuMeasurements imuIntegratorOpt_;

  bool systemInitialized;

  std::deque<sensor_msgs::Imu> imuQueOpt;
  std::deque<sensor_msgs::Imu> imuQueImu;

  gtsam::Pose3 prevPose_;
  gtsam::Vector3 prevVel_;
  gtsam::NavState prevState_;
  gtsam::imuBias::ConstantBias prevBias_;

  gtsam::NavState prev_odom_;
  gtsam::imuBias::ConstantBias prev_odom_bias_;

  bool doneFirstOpt = false;
  double lastImuT_imu = -1;
  double lastImuT_opt = -1;

  gtsam::ISAM2 optimizer;

  const double delta_t = 0;

  int key = 1;

  gtsam::Pose3 imu2Lidar = gtsam::Pose3(
    gtsam::Rot3(1, 0, 0, 0),
    gtsam::Point3(-extTrans.x(), -extTrans.y(), -extTrans.z()));
  gtsam::Pose3 lidar2Imu = gtsam::Pose3(
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
    noiseModelBetweenBias(
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

    const double currentCorrectionTime = timeInSec(odom_msg->header);

    // make sure we have imu data to integrate
    if (imuQueOpt.empty()) {
      return;
    }

    gtsam::Pose3 lidar_pose = makeGtsamPose(odom_msg->pose.pose);

    // 0. initialize system
    if (!systemInitialized) {
      // pop old IMU message
      while (!imuQueOpt.empty()) {
        if (timeInSec(imuQueOpt.front().header) >= currentCorrectionTime - delta_t) {
          break;
        }
        lastImuT_opt = timeInSec(imuQueOpt.front().header);
        imuQueOpt.pop_front();
      }

      optimizer = initOptimizer(lidar2Imu, lidar_pose);

      imuIntegratorImu_.resetIntegrationAndSetBias(prevBias_);
      imuIntegratorOpt_.resetIntegrationAndSetBias(prevBias_);

      key = 1;
      systemInitialized = true;
      return;
    }

    // reset graph for speed
    if (key == 100) {
      resetOptimizer(prevPose_, prevVel_, prevBias_, key, optimizer);

      key = 1;
    }

    // 1. integrate imu data and optimize
    while (!imuQueOpt.empty()) {
      // pop and integrate imu data that is between two optimizations
      sensor_msgs::Imu & front = imuQueOpt.front();
      const double imuTime = timeInSec(front.header);
      if (imuTime >= currentCorrectionTime - delta_t) {
        break;
      }
      const double dt = (lastImuT_opt < 0) ? (1.0 / 500.0) : (imuTime - lastImuT_opt);

      imuIntegratorOpt_.integrateMeasurement(
        vector3ToEigen(front.linear_acceleration),
        vector3ToEigen(front.angular_velocity),
        dt
      );

      lastImuT_opt = imuTime;
      imuQueOpt.pop_front();
    }

    // add imu factor to graph
    const gtsam::PreintegratedImuMeasurements & preint_imu =
      dynamic_cast<const gtsam::PreintegratedImuMeasurements &>(imuIntegratorOpt_);
    gtsam::ImuFactor imu_factor(X(key - 1), V(key - 1), X(key), V(key), B(key - 1), preint_imu);
    gtsam::NonlinearFactorGraph graphFactors;

    graphFactors.add(imu_factor);
    // add imu bias between factor
    graphFactors.add(
      gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(
        B(key - 1), B(key), gtsam::imuBias::ConstantBias(),
        Diagonal::Sigmas(sqrt(imuIntegratorOpt_.deltaTij()) * noiseModelBetweenBias))
    );

    const Diagonal::shared_ptr correctionNoise(
      Diagonal::Sigmas((Vector6d() << 0.05, 0.05, 0.05, 0.1, 0.1, 0.1).finished())
    );
    const Diagonal::shared_ptr correctionNoise2(Diagonal::Sigmas(Vector6d::Ones()));

    const auto noise = odom_msg->pose.covariance[0] == 1 ? correctionNoise2 : correctionNoise;
    // add pose factor
    const gtsam::Pose3 curr_imu_pose = lidar_pose.compose(lidar2Imu);
    const gtsam::PriorFactor<gtsam::Pose3> pose_factor(X(key), curr_imu_pose, noise);
    graphFactors.add(pose_factor);
    // insert predicted values
    const gtsam::NavState state = imuIntegratorOpt_.predict(prevState_, prevBias_);
    gtsam::Values graphValues;
    graphValues.insert(X(key), state.pose());
    graphValues.insert(V(key), state.v());
    graphValues.insert(B(key), prevBias_);
    // optimize
    optimizer.update(graphFactors, graphValues);
    optimizer.update();
    // Overwrite the beginning of the preintegration for the next step.
    const gtsam::Values result = optimizer.calculateEstimate();
    prevPose_ = result.at<gtsam::Pose3>(X(key));
    prevVel_ = result.at<gtsam::Vector3>(V(key));
    prevState_ = gtsam::NavState(prevPose_, prevVel_);
    prevBias_ = result.at<gtsam::imuBias::ConstantBias>(B(key));
    // Reset the optimization preintegration object.
    imuIntegratorOpt_.resetIntegrationAndSetBias(prevBias_);
    // check optimization
    if (failureDetection(prevVel_, prevBias_)) {
      lastImuT_imu = -1;
      doneFirstOpt = false;
      systemInitialized = false;
      return;
    }

    // 2. after optiization, re-propagate imu odometry preintegration
    prev_odom_ = prevState_;
    prev_odom_bias_ = prevBias_;
    // first pop imu message older than current correction data
    double lastImuQT = -1;
    while (!imuQueImu.empty() &&
      timeInSec(imuQueImu.front().header) < currentCorrectionTime - delta_t)
    {
      lastImuQT = timeInSec(imuQueImu.front().header);
      imuQueImu.pop_front();
    }
    // repropogate
    if (!imuQueImu.empty()) {
      // reset bias use the newly optimized bias
      imuIntegratorImu_.resetIntegrationAndSetBias(prev_odom_bias_);
      // integrate imu message from the beginning of this optimization
      for (unsigned int i = 0; i < imuQueImu.size(); ++i) {
        const sensor_msgs::Imu & msg = imuQueImu[i];
        const double imuTime = timeInSec(msg.header);
        const double dt = (lastImuQT < 0) ? (1.0 / 500.0) : (imuTime - lastImuQT);
        imuIntegratorImu_.integrateMeasurement(
          vector3ToEigen(msg.linear_acceleration),
          vector3ToEigen(msg.angular_velocity),
          dt
        );
        lastImuQT = imuTime;
      }
    }

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
    imuQueImu.push_back(imu);

    if (!doneFirstOpt) {
      return;
    }

    const double imuTime = timeInSec(imu.header);
    const double dt = (lastImuT_imu < 0) ? (1.0 / 500.0) : (imuTime - lastImuT_imu);
    lastImuT_imu = imuTime;

    const Eigen::Vector3d linear_acceleration = vector3ToEigen(imu.linear_acceleration);
    const Eigen::Vector3d angular_velocity = vector3ToEigen(imu.angular_velocity);
    imuIntegratorImu_.integrateMeasurement(linear_acceleration, angular_velocity, dt);

    // predict odometry
    const gtsam::NavState current_imu = imuIntegratorImu_.predict(prev_odom_, prev_odom_bias_);

    // publish odometry
    nav_msgs::Odometry odometry;
    odometry.header.stamp = imu.header.stamp;
    odometry.header.frame_id = odometryFrame;
    odometry.child_frame_id = "odom_imu";

    // transform imu pose to ldiar
    const gtsam::Pose3 lidar_pose = current_imu.pose().compose(imu2Lidar);

    odometry.pose.pose = makePose(
      eigenToQuaternion(lidar_pose.rotation().toQuaternion()),
      eigenToPoint(lidar_pose.translation())
    );

    odometry.twist.twist = makeTwist(
      eigenToVector3(angular_velocity + prev_odom_bias_.gyroscope()),
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
