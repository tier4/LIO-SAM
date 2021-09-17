#include "utility.h"

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

class TransformFusion : public ParamServer
{
public:
  std::mutex mtx;

  ros::Subscriber subImuOdometry;
  ros::Subscriber subLaserOdometry;

  ros::Publisher pubImuOdometry;
  ros::Publisher pubImuPath;

  Eigen::Affine3d lidarOdomAffine;
  Eigen::Affine3d imuOdomAffineFront;
  Eigen::Affine3d imuOdomAffineBack;

  tf::TransformListener tfListener;
  tf::StampedTransform lidar2Baselink;

  double lidarOdomTime = -1;
  std::deque<nav_msgs::Odometry> imuOdomQueue;

  TransformFusion()
  {
    if (lidarFrame != baselinkFrame) {
      try {
        tfListener.waitForTransform(
          lidarFrame, baselinkFrame, ros::Time(0),
          ros::Duration(3.0));
        tfListener.lookupTransform(
          lidarFrame, baselinkFrame, ros::Time(0),
          lidar2Baselink);
      } catch (tf::TransformException ex) {
        ROS_ERROR("%s", ex.what());
      }
    }

    subLaserOdometry = nh.subscribe<nav_msgs::Odometry>(
      "lio_sam/mapping/odometry",
      5, &TransformFusion::lidarOdometryHandler, this,
      ros::TransportHints().tcpNoDelay());
    subImuOdometry = nh.subscribe<nav_msgs::Odometry>(
      odomTopic + "_incremental",
      2000, &TransformFusion::imuOdometryHandler, this,
      ros::TransportHints().tcpNoDelay());

    pubImuOdometry = nh.advertise<nav_msgs::Odometry>(odomTopic, 2000);
    pubImuPath = nh.advertise<nav_msgs::Path>("lio_sam/imu/path", 1);
  }

  void lidarOdometryHandler(const nav_msgs::Odometry::ConstPtr & odomMsg)
  {
    std::lock_guard<std::mutex> lock(mtx);

    lidarOdomAffine = poseToAffine(odomMsg->pose.pose);

    lidarOdomTime = odomMsg->header.stamp.toSec();
  }

  void imuOdometryHandler(const nav_msgs::Odometry::ConstPtr & odomMsg)
  {
    // static tf
    static tf::TransformBroadcaster tfMap2Odom;
    static tf::Transform map_to_odom = tf::Transform(
      tf::createQuaternionFromRPY(
        0,
        0, 0), tf::Vector3(0, 0, 0));
    tfMap2Odom.sendTransform(
      tf::StampedTransform(
        map_to_odom,
        odomMsg->header.stamp, mapFrame, odometryFrame));

    std::lock_guard<std::mutex> lock(mtx);

    imuOdomQueue.push_back(*odomMsg);

    // get latest odometry (at current IMU stamp)
    if (lidarOdomTime == -1) {
      return;
    }

    dropBefore(lidarOdomTime, imuOdomQueue);
    Eigen::Affine3d imuOdomAffineFront = poseToAffine(imuOdomQueue.front().pose.pose);
    Eigen::Affine3d imuOdomAffineBack = poseToAffine(imuOdomQueue.back().pose.pose);
    Eigen::Affine3d imuOdomAffineIncre = imuOdomAffineFront.inverse() *
      imuOdomAffineBack;
    Eigen::Affine3d imuOdomAffineLast = lidarOdomAffine * imuOdomAffineIncre;

    // publish latest odometry
    nav_msgs::Odometry laserOdometry = imuOdomQueue.back();
    laserOdometry.pose.pose = affineToPose(imuOdomAffineLast);
    pubImuOdometry.publish(laserOdometry);

    // publish tf
    static tf::TransformBroadcaster tfOdom2BaseLink;
    tf::Transform tCur;
    tf::poseMsgToTF(laserOdometry.pose.pose, tCur);
    if (lidarFrame != baselinkFrame) {
      tCur = tCur * lidar2Baselink;
    }
    tf::StampedTransform odom_2_baselink = tf::StampedTransform(
      tCur,
      odomMsg->header.stamp, odometryFrame, baselinkFrame);
    tfOdom2BaseLink.sendTransform(odom_2_baselink);

    // publish IMU path
    static nav_msgs::Path imuPath;
    static double last_path_time = -1;
    double imuTime = imuOdomQueue.back().header.stamp.toSec();
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

class IMUPreintegration : public ParamServer
{
public:
  std::mutex mtx;

  ros::Subscriber subImu;
  ros::Subscriber subOdometry;
  ros::Publisher pubImuOdometry;

  bool systemInitialized = false;

  gtsam::noiseModel::Diagonal::shared_ptr priorPoseNoise;
  gtsam::noiseModel::Diagonal::shared_ptr priorVelNoise;
  gtsam::noiseModel::Diagonal::shared_ptr priorBiasNoise;
  gtsam::noiseModel::Diagonal::shared_ptr correctionNoise;
  gtsam::noiseModel::Diagonal::shared_ptr correctionNoise2;
  gtsam::Vector noiseModelBetweenBias;


  gtsam::PreintegratedImuMeasurements * imuIntegratorOpt_;
  gtsam::PreintegratedImuMeasurements * imuIntegratorImu_;

  std::deque<sensor_msgs::Imu> imuQueOpt;
  std::deque<sensor_msgs::Imu> imuQueImu;

  gtsam::Pose3 prevPose_;
  gtsam::Vector3 prevVel_;
  gtsam::NavState prevState_;
  gtsam::imuBias::ConstantBias prevBias_;

  gtsam::NavState prev_;
  gtsam::imuBias::ConstantBias prev_bias_;

  bool doneFirstOpt = false;
  double lastImuT_imu = -1;
  double lastImuT_opt = -1;

  gtsam::ISAM2 optimizer;
  gtsam::NonlinearFactorGraph graphFactors;
  gtsam::Values graphValues;

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
  {
    subImu = nh.subscribe<sensor_msgs::Imu>(
      imuTopic, 2000, &IMUPreintegration::imuHandler,
      this, ros::TransportHints().tcpNoDelay());
    subOdometry = nh.subscribe<nav_msgs::Odometry>(
      "lio_sam/mapping/odometry_incremental", 5,
      &IMUPreintegration::odometryHandler,
      this, ros::TransportHints().tcpNoDelay());
    pubImuOdometry = nh.advertise<nav_msgs::Odometry>(
      odomTopic + "_incremental",
      2000);

    boost::shared_ptr<gtsam::PreintegrationParams> p =
      gtsam::PreintegrationParams::MakeSharedU(imuGravity);
    p->accelerometerCovariance = gtsam::Matrix33::Identity(3, 3) * pow(
      imuAccNoise,
      2);                             // acc white noise in continuous
    p->gyroscopeCovariance = gtsam::Matrix33::Identity(3, 3) * pow(
      imuGyrNoise,
      2);                             // gyro white noise in continuous
    p->integrationCovariance = gtsam::Matrix33::Identity(3, 3) * pow(
      1e-4,
      2);                             // error committed in integrating position from velocities
    gtsam::imuBias::ConstantBias prior_imu_bias((gtsam::Vector(6) << 0, 0, 0, 0, 0,
      0).finished());    // assume zero initial bias

    // rad,rad,rad,m, m, m
    priorPoseNoise = gtsam::noiseModel::Diagonal::Sigmas(
      (gtsam::Vector(6) << 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2).finished()
    );
    priorVelNoise = gtsam::noiseModel::Isotropic::Sigma(3, 1e4);   // m/s
    // 1e-2 ~ 1e-3 seems to be good
    priorBiasNoise = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3);
    // rad,rad,rad,m, m, m
    correctionNoise = gtsam::noiseModel::Diagonal::Sigmas(
      (gtsam::Vector(6) << 0.05, 0.05, 0.05, 0.1, 0.1, 0.1).finished()
    );
    // rad,rad,rad,m, m, m
    correctionNoise2 = gtsam::noiseModel::Diagonal::Sigmas(
      (gtsam::Vector(6) << 1, 1, 1, 1, 1, 1).finished()
    );
    noiseModelBetweenBias = \
      (gtsam::Vector(6) << imuAccBiasN, imuAccBiasN, imuAccBiasN,
      imuGyrBiasN, imuGyrBiasN, imuGyrBiasN).finished();

    // setting up the IMU integration for IMU message thread
    imuIntegratorImu_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias);
    // setting up the IMU integration for optimization
    imuIntegratorOpt_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias);
  }

  void resetOptimization()
  {
    gtsam::ISAM2Params optParameters;
    optParameters.relinearizeThreshold = 0.1;
    optParameters.relinearizeSkip = 1;
    optimizer = gtsam::ISAM2(optParameters);

    gtsam::NonlinearFactorGraph newGraphFactors;
    graphFactors = newGraphFactors;

    gtsam::Values NewGraphValues;
    graphValues = NewGraphValues;
  }

  void resetParams()
  {
    lastImuT_imu = -1;
    doneFirstOpt = false;
    systemInitialized = false;
  }

  void odometryHandler(const nav_msgs::Odometry::ConstPtr & odomMsg)
  {
    std::lock_guard<std::mutex> lock(mtx);

    double currentCorrectionTime = ROS_TIME(odomMsg);

    // make sure we have imu data to integrate
    if (imuQueOpt.empty()) {
      return;
    }

    gtsam::Pose3 lidarPose = makeGtsamPose(odomMsg->pose.pose);
    bool degenerate = (int)odomMsg->pose.covariance[0] == 1 ? true : false;

    // 0. initialize system
    if (systemInitialized == false) {
      resetOptimization();

      // pop old IMU message
      while (!imuQueOpt.empty()) {
        if (ROS_TIME(&imuQueOpt.front()) < currentCorrectionTime - delta_t) {
          lastImuT_opt = ROS_TIME(&imuQueOpt.front());
          imuQueOpt.pop_front();
        } else {
          break;
        }
      }
      // initial pose
      prevPose_ = lidarPose.compose(lidar2Imu);
      gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, priorPoseNoise);
      graphFactors.add(priorPose);
      // initial velocity
      prevVel_ = gtsam::Vector3(0, 0, 0);
      gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, priorVelNoise);
      graphFactors.add(priorVel);
      // initial bias
      prevBias_ = gtsam::imuBias::ConstantBias();
      gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_,
        priorBiasNoise);
      graphFactors.add(priorBias);
      // add values
      graphValues.insert(X(0), prevPose_);
      graphValues.insert(V(0), prevVel_);
      graphValues.insert(B(0), prevBias_);
      // optimize once
      optimizer.update(graphFactors, graphValues);
      graphFactors.resize(0);
      graphValues.clear();

      imuIntegratorImu_->resetIntegrationAndSetBias(prevBias_);
      imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);

      key = 1;
      systemInitialized = true;
      return;
    }


    // reset graph for speed
    if (key == 100) {
      // get updated noise before reset
      gtsam::noiseModel::Gaussian::shared_ptr updatedPoseNoise =
        gtsam::noiseModel::Gaussian::Covariance(
        optimizer.marginalCovariance(
          X(
            key - 1)));
      gtsam::noiseModel::Gaussian::shared_ptr updatedVelNoise =
        gtsam::noiseModel::Gaussian::Covariance(
        optimizer.marginalCovariance(
          V(
            key - 1)));
      gtsam::noiseModel::Gaussian::shared_ptr updatedBiasNoise =
        gtsam::noiseModel::Gaussian::Covariance(
        optimizer.marginalCovariance(
          B(
            key - 1)));
      // reset graph
      resetOptimization();
      // add pose
      gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, updatedPoseNoise);
      graphFactors.add(priorPose);
      // add velocity
      gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, updatedVelNoise);
      graphFactors.add(priorVel);
      // add bias
      gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_,
        updatedBiasNoise);
      graphFactors.add(priorBias);
      // add values
      graphValues.insert(X(0), prevPose_);
      graphValues.insert(V(0), prevVel_);
      graphValues.insert(B(0), prevBias_);
      // optimize once
      optimizer.update(graphFactors, graphValues);
      graphFactors.resize(0);
      graphValues.clear();

      key = 1;
    }

    // 1. integrate imu data and optimize
    while (!imuQueOpt.empty()) {
      // pop and integrate imu data that is between two optimizations
      sensor_msgs::Imu * thisImu = &imuQueOpt.front();
      double imuTime = ROS_TIME(thisImu);
      if (imuTime < currentCorrectionTime - delta_t) {
        const double dt = (lastImuT_opt < 0) ? (1.0 / 500.0) : (imuTime - lastImuT_opt);

        const geometry_msgs::Vector3 a = thisImu->linear_acceleration;
        const geometry_msgs::Vector3 w = thisImu->angular_velocity;
        imuIntegratorOpt_->integrateMeasurement(
          gtsam::Vector3(a.x, a.y, a.z),
          gtsam::Vector3(w.x, w.y, w.z),
          dt
        );

        lastImuT_opt = imuTime;
        imuQueOpt.pop_front();
      } else {
        break;
      }
    }
    // add imu factor to graph
    const gtsam::PreintegratedImuMeasurements & preint_imu =
      dynamic_cast<const gtsam::PreintegratedImuMeasurements &>(*imuIntegratorOpt_);
    gtsam::ImuFactor imu_factor(X(key - 1), V(key - 1), X(key), V(key), B(key - 1),
      preint_imu);
    graphFactors.add(imu_factor);
    // add imu bias between factor
    graphFactors.add(
      gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(
        B(key - 1),
        B(key), gtsam::imuBias::ConstantBias(),
        gtsam::noiseModel::Diagonal::Sigmas(
          sqrt(imuIntegratorOpt_->deltaTij()) *
          noiseModelBetweenBias)));
    // add pose factor
    gtsam::Pose3 curPose = lidarPose.compose(lidar2Imu);
    gtsam::PriorFactor<gtsam::Pose3> pose_factor(X(key), curPose,
      degenerate ? correctionNoise2 : correctionNoise);
    graphFactors.add(pose_factor);
    // insert predicted values
    gtsam::NavState propState_ = imuIntegratorOpt_->predict(prevState_, prevBias_);
    graphValues.insert(X(key), propState_.pose());
    graphValues.insert(V(key), propState_.v());
    graphValues.insert(B(key), prevBias_);
    // optimize
    optimizer.update(graphFactors, graphValues);
    optimizer.update();
    graphFactors.resize(0);
    graphValues.clear();
    // Overwrite the beginning of the preintegration for the next step.
    gtsam::Values result = optimizer.calculateEstimate();
    prevPose_ = result.at<gtsam::Pose3>(X(key));
    prevVel_ = result.at<gtsam::Vector3>(V(key));
    prevState_ = gtsam::NavState(prevPose_, prevVel_);
    prevBias_ = result.at<gtsam::imuBias::ConstantBias>(B(key));
    // Reset the optimization preintegration object.
    imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);
    // check optimization
    if (failureDetection(prevVel_, prevBias_)) {
      resetParams();
      return;
    }


    // 2. after optiization, re-propagate imu odometry preintegration
    prev_ = prevState_;
    prev_bias_ = prevBias_;
    // first pop imu message older than current correction data
    double lastImuQT = -1;
    while (!imuQueImu.empty() &&
      ROS_TIME(&imuQueImu.front()) < currentCorrectionTime - delta_t)
    {
      lastImuQT = ROS_TIME(&imuQueImu.front());
      imuQueImu.pop_front();
    }
    // repropogate
    if (!imuQueImu.empty()) {
      // reset bias use the newly optimized bias
      imuIntegratorImu_->resetIntegrationAndSetBias(prev_bias_);
      // integrate imu message from the beginning of this optimization
      for (int i = 0; i < (int)imuQueImu.size(); ++i) {
        sensor_msgs::Imu * thisImu = &imuQueImu[i];
        double imuTime = ROS_TIME(thisImu);

        const double dt = (lastImuQT < 0) ? (1.0 / 500.0) : (imuTime - lastImuQT);
        const geometry_msgs::Vector3 a = thisImu->linear_acceleration;
        const geometry_msgs::Vector3 w = thisImu->angular_velocity;
        imuIntegratorImu_->integrateMeasurement(
          gtsam::Vector3(a.x, a.y, a.z),
          gtsam::Vector3(w.x, w.y, w.z),
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

    const sensor_msgs::Imu thisImu = [&] {
        try {
          return imu_converter_.imuConverter(*imu_raw);
        } catch (const std::runtime_error & e) {
          ROS_ERROR(e.what());
          ros::shutdown();
          return sensor_msgs::Imu();
        }
      } ();

    imuQueOpt.push_back(thisImu);
    imuQueImu.push_back(thisImu);

    if (doneFirstOpt == false) {
      return;
    }

    const double imuTime = timeInSec(thisImu.header);
    const double dt = (lastImuT_imu < 0) ? (1.0 / 500.0) : (imuTime - lastImuT_imu);
    lastImuT_imu = imuTime;

    const Eigen::Vector3d linear_acceleration = vector3ToEigen(thisImu.linear_acceleration);
    const Eigen::Vector3d angular_velocity = vector3ToEigen(thisImu.angular_velocity);
    imuIntegratorImu_->integrateMeasurement(linear_acceleration, angular_velocity, dt);

    // predict odometry
    const gtsam::NavState current_imu = imuIntegratorImu_->predict(prev_, prev_bias_);

    // publish odometry
    nav_msgs::Odometry odometry;
    odometry.header.stamp = thisImu.header.stamp;
    odometry.header.frame_id = odometryFrame;
    odometry.child_frame_id = "odom_imu";

    // transform imu pose to ldiar
    const gtsam::Pose3 lidar_pose = current_imu.pose().compose(imu2Lidar);

    odometry.pose.pose.position = eigenToPoint(lidar_pose.translation());
    odometry.pose.pose.orientation = eigenToQuaternion(lidar_pose.rotation().toQuaternion());

    odometry.twist.twist.linear = eigenToVector3(current_imu.velocity());
    const Eigen::Vector3d w = prev_bias_.gyroscope();
    const Eigen::Vector3d v = angular_velocity;
    odometry.twist.twist.angular = eigenToVector3(v + w);
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
