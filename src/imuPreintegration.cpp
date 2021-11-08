#include "utility.hpp"
#include "param_server.h"
#include "transform_fusion.hpp"

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

gtsam::ISAM2 initOptimizer(const gtsam::Pose3 & pose)
{
  const gtsam::ISAM2Params params(gtsam::ISAM2GaussNewtonParams(), 0.1, 1);

  gtsam::NonlinearFactorGraph graph;

  const Diagonal::shared_ptr pose_noise(Diagonal::Sigmas(1e-2 * Vector6d::Ones()));
  // rad,rad,rad, m, m, m (m/s)
  const Diagonal::shared_ptr velocity_noise(gtsam::noiseModel::Isotropic::Sigma(3, 1e4));
  // 1e-2 ~ 1e-3 seems to be good
  const Diagonal::shared_ptr bias_noise(gtsam::noiseModel::Isotropic::Sigma(6, 1e-3));

  graph.add(gtsam::PriorFactor<gtsam::Pose3>(P(0), pose, pose_noise));

  const gtsam::Vector3 velocity = gtsam::Vector3(0, 0, 0);
  graph.add(gtsam::PriorFactor<gtsam::Vector3>(V(0), velocity, velocity_noise));

  const gtsam::imuBias::ConstantBias bias = gtsam::imuBias::ConstantBias();
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
  const boost::shared_ptr<gtsam::PreintegrationParams> & integration_params_,
  gtsam::PreintegratedImuMeasurements & integrator,
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
  integrator = gtsam::PreintegratedImuMeasurements(integration_params_, prev_odom_bias_);
  // integrate imu message from the beginning of this optimization
  for (unsigned int i = 0; i < imu_queue.size(); ++i) {
    const sensor_msgs::Imu & msg = imu_queue[i];
    const double imu_time = timeInSec(msg.header);
    const double dt = (last_imu_time < 0) ? (1.0 / 500.0) : (imu_time - last_imu_time);

    integrator.integrateMeasurement(
      vector3ToEigen(msg.linear_acceleration),
      vector3ToEigen(msg.angular_velocity),
      dt
    );

    last_imu_time = imu_time;
  }
}

void imuIntegration(
  const double time_threshold, double & last_imu_time,
  gtsam::PreintegratedImuMeasurements & integrator,
  std::deque<sensor_msgs::Imu> & imu_queue)
{
  while (!imu_queue.empty() && timeInSec(imu_queue.front().header) < time_threshold) {
    // pop and integrate imu data that is between two optimizations
    const sensor_msgs::Imu & front = imu_queue.front();
    const double imu_time = timeInSec(front.header);
    const double dt = (last_imu_time < 0) ? (1.0 / 500.0) : (imu_time - last_imu_time);

    integrator.integrateMeasurement(
      vector3ToEigen(front.linear_acceleration),
      vector3ToEigen(front.angular_velocity),
      dt
    );

    last_imu_time = imu_time;
    imu_queue.pop_front();
  }
}

Diagonal::shared_ptr getCovariance(const bool is_degenerate)
{
  if (is_degenerate) {
    return Diagonal::Sigmas(Vector6d::Ones());
  }
  return Diagonal::Sigmas((Vector6d() << 0.05, 0.05, 0.05, 0.1, 0.1, 0.1).finished());
}

class IMUPreintegration : public ParamServer
{
public:
  std::mutex mtx;

  const ros::Subscriber imu_subscriber_;
  const ros::Subscriber incremental_odometry_subscriber_;
  const ros::Publisher imu_incremental_odometry_publisher_;

  const gtsam::Pose3 imu_to_lidar;
  const gtsam::Pose3 lidar_to_imu;

  const boost::shared_ptr<gtsam::PreintegrationParams> integration_params_;
  const gtsam::imuBias::ConstantBias prior_imu_bias_;

  const gtsam::Vector between_noise_bias_;

  gtsam::PreintegratedImuMeasurements integrator_;

  bool systemInitialized;

  std::deque<sensor_msgs::Imu> imuQueOpt;
  std::deque<sensor_msgs::Imu> imu_queue;

  gtsam::NavState prev_state_;
  gtsam::imuBias::ConstantBias prev_bias_;

  bool doneFirstOpt = false;
  double last_imu_time = -1;
  double last_imu_time_opt = -1;

  gtsam::ISAM2 optimizer;

  const double delta_t = 0;

  int key = 1;

  const IMUConverter imu_converter_;

  IMUPreintegration()
  : imu_subscriber_(
      nh.subscribe<sensor_msgs::Imu>(
        imuTopic, 2000, &IMUPreintegration::imuHandler,
        this, ros::TransportHints().tcpNoDelay())),
    incremental_odometry_subscriber_(
      nh.subscribe<nav_msgs::Odometry>(
        "lio_sam/mapping/odometry_incremental", 5, &IMUPreintegration::odometryHandler,
        this, ros::TransportHints().tcpNoDelay())),
    imu_incremental_odometry_publisher_(
      nh.advertise<geometry_msgs::TransformStamped>(imu_incremental_odometry_topic, 2000)),
    imu_to_lidar(
      gtsam::Pose3(
        gtsam::Rot3(1, 0, 0, 0),
        gtsam::Point3(-extTrans.x(), -extTrans.y(), -extTrans.z()))),
    lidar_to_imu(
      gtsam::Pose3(
        gtsam::Rot3(1, 0, 0, 0),
        gtsam::Point3(extTrans.x(), extTrans.y(), extTrans.z()))),
    integration_params_(initialIntegrationParams(imuGravity, imuAccNoise, imuGyrNoise)),
    prior_imu_bias_(Vector6d::Zero()),
    between_noise_bias_(
      (Vector6d() <<
        imuAccBiasN, imuAccBiasN, imuAccBiasN,
        imuGyrBiasN, imuGyrBiasN, imuGyrBiasN).finished()),
    integrator_(gtsam::PreintegratedImuMeasurements(integration_params_, prior_imu_bias_)),
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

    auto imuIntegratorOpt_ = gtsam::PreintegratedImuMeasurements(integration_params_, prev_bias_);

    // 0. initialize system
    if (!systemInitialized) {
      // pop old IMU message
      popOldMessages(odom_time - delta_t, last_imu_time_opt, imuQueOpt);

      optimizer = initOptimizer(lidar_pose.compose(lidar_to_imu));

      key = 1;
      systemInitialized = true;
      return;
    }

    // 1. integrate imu data and optimize
    imuIntegration(odom_time - delta_t, last_imu_time_opt, imuIntegratorOpt_, imuQueOpt);

    gtsam::NonlinearFactorGraph graph;

    const bool is_degenerate = odom_msg->pose.covariance[0] == 1;
    const auto noise = getCovariance(is_degenerate);
    graph.add(gtsam::PriorFactor<gtsam::Pose3>(P(key), lidar_pose.compose(lidar_to_imu), noise));

    graph.add(
      gtsam::ImuFactor(P(key - 1), V(key - 1), P(key), V(key), B(key - 1), imuIntegratorOpt_));

    graph.add(
      gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(
        B(key - 1), B(key), gtsam::imuBias::ConstantBias(),
        Diagonal::Sigmas(sqrt(imuIntegratorOpt_.deltaTij()) * between_noise_bias_))
    );

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
    const gtsam::Pose3 pose = result.at<gtsam::Pose3>(P(key));
    const gtsam::Vector3 velocity = result.at<gtsam::Vector3>(V(key));
    prev_state_ = gtsam::NavState(pose, velocity);
    prev_bias_ = result.at<gtsam::imuBias::ConstantBias>(B(key));
    // check optimization
    if (failureDetection(velocity, prev_bias_)) {
      last_imu_time = -1;
      doneFirstOpt = false;
      systemInitialized = false;
      return;
    }

    // 2. after optiization, re-propagate imu odometry preintegration
    imuPreIntegration(odom_time - delta_t, prev_bias_, integration_params_, integrator_, imu_queue);

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
    integrator_.integrateMeasurement(linear_acceleration, angular_velocity, dt);

    // predict odometry
    const gtsam::NavState current_imu = integrator_.predict(prev_state_, prev_bias_);

    imu_incremental_odometry_publisher_.publish(
      makeTransformStamped(
        imu.header.stamp, odometryFrame, "odom_imu",
        makeTransform(current_imu.pose().compose(imu_to_lidar)))
    );
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
