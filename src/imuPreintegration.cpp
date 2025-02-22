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

#include "range/v3/all.hpp"

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

bool validateVelocity(const Eigen::Vector3d & velocity)
{
  return velocity.norm() <= 30;
}

bool validateBias(const gtsam::imuBias::ConstantBias & bias)
{
  const Eigen::Vector3d ba = bias.accelerometer();
  const Eigen::Vector3d bg = bias.gyroscope();
  return ba.norm() <= 1.0 && bg.norm() <= 1.0;
}

using Diagonal = gtsam::noiseModel::Diagonal;

gtsam::ISAM2 initOptimizer(const gtsam::Pose3 & pose)
{
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

  const gtsam::ISAM2Params params(gtsam::ISAM2GaussNewtonParams(), 0.1, 1);
  gtsam::ISAM2 optimizer = gtsam::ISAM2(params);
  optimizer.update(graph, values);
  return optimizer;
}

std::tuple<gtsam::PreintegratedImuMeasurements, double>
makeIntegrator(
  const boost::shared_ptr<gtsam::PreintegrationParams> & params,
  const gtsam::imuBias::ConstantBias & bias,
  const double last_imu_time,
  const std::vector<sensor_msgs::Imu> & imus)
{
  auto integrator = gtsam::PreintegratedImuMeasurements(params, bias);

  double last = last_imu_time;

  for (const sensor_msgs::Imu & imu : imus) {
    const double imu_time = timeInSec(imu.header);

    integrator.integrateMeasurement(
      vector3ToEigen(imu.linear_acceleration),
      vector3ToEigen(imu.angular_velocity),
      imu_time - last
    );

    last = imu_time;
  }
  return {integrator, last};
}

Diagonal::shared_ptr getCovariance(const bool is_degenerate)
{
  if (is_degenerate) {
    return Diagonal::Sigmas(Vector6d::Ones());
  }
  return Diagonal::Sigmas((Vector6d() << 0.05, 0.05, 0.05, 0.1, 0.1, 0.1).finished());
}

class StatePrediction
{
public:
  StatePrediction(const gtsam::Pose3 & pose)
  : optimizer(initOptimizer(pose)),
    prev_pose_(gtsam::Pose3::identity()),
    prev_velocity_(gtsam::Vector3::Zero()),
    prev_bias_(gtsam::imuBias::ConstantBias::identity())
  {
  }

  StatePrediction() {}

  std::tuple<gtsam::Pose3, gtsam::Vector3, gtsam::imuBias::ConstantBias> update(
    const int key,
    const gtsam::PreintegratedImuMeasurements & imu_integrator,
    const gtsam::NonlinearFactorGraph & graph)
  {
    const gtsam::NavState state = imu_integrator.predict(
      gtsam::NavState(prev_pose_, prev_velocity_), prev_bias_);
    gtsam::Values values;
    values.insert(P(key), state.pose());
    values.insert(V(key), state.v());
    values.insert(B(key), prev_bias_);

    // optimize
    optimizer.update(graph, values);
    optimizer.update();

    const gtsam::Values result = optimizer.calculateEstimate();
    const gtsam::Pose3 pose = result.at<gtsam::Pose3>(P(key));
    const gtsam::Vector3 velocity = result.at<gtsam::Vector3>(V(key));
    const gtsam::imuBias::ConstantBias bias = result.at<gtsam::imuBias::ConstantBias>(B(key));
    prev_pose_ = pose;
    prev_velocity_ = velocity;
    prev_bias_ = bias;
    return {pose, velocity, bias};
  }

private:
  gtsam::ISAM2 optimizer;
  gtsam::Pose3 prev_pose_;
  gtsam::Vector3 prev_velocity_;
  gtsam::imuBias::ConstantBias prev_bias_;
};

gtsam::PriorFactor<gtsam::Pose3> makePrior(
  const int key,
  const gtsam::Pose3 & imu_pose,
  const gtsam::SharedNoiseModel & noise)
{
  return gtsam::PriorFactor<gtsam::Pose3>(P(key), imu_pose, noise);
}

gtsam::ImuFactor makeImuConstraint(
  const int key,
  const gtsam::PreintegratedImuMeasurements & integrator)
{
  return gtsam::ImuFactor(P(key - 1), V(key - 1), P(key), V(key), B(key - 1), integrator);
}

gtsam::BetweenFactor<gtsam::imuBias::ConstantBias> makeBiasConstraint(
  const int key, const double dt,
  const Vector6d & between_noise_bias)
{
  return gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(
    B(key - 1), B(key), gtsam::imuBias::ConstantBias(),
    Diagonal::Sigmas(sqrt(dt) * between_noise_bias));
}

class ImuIntegratorFactory
{
private:
  double last_imu_time_ = -1;
  std::deque<sensor_msgs::Imu> imu_queue;
  gtsam::PreintegratedImuMeasurements integrator_;
  const boost::shared_ptr<gtsam::PreintegrationParams> params_;

public:
  ImuIntegratorFactory(
    const boost::shared_ptr<gtsam::PreintegrationParams> integration_params)
  : params_(integration_params) {}

  bool isAvailable() const
  {
    return !imu_queue.empty();
  }

  void init(const double lidar_time)
  {
    popOldMessages(lidar_time, last_imu_time_, imu_queue);
  }

  void push(const sensor_msgs::Imu & imu)
  {
    imu_queue.push_back(imu);
  }

  void build(const gtsam::imuBias::ConstantBias & bias_, const double lidar_time)
  {
    const auto f = [&](const sensor_msgs::Imu & imu) {return timeInSec(imu.header) < lidar_time;};
    const auto imus = imu_queue | ranges::views::filter(f) | ranges::to_vector;
    std::tie(integrator_, last_imu_time_) = makeIntegrator(params_, bias_, last_imu_time_, imus);
    while (!imu_queue.empty() && timeInSec(imu_queue.front().header) < lidar_time) {
      imu_queue.pop_front();
    }
  }

  gtsam::PreintegratedImuMeasurements get() const
  {
    return integrator_;
  }
};

class IMUPreintegration : public ParamServer
{
public:
  std::mutex mtx;

  const ros::Subscriber imu_subscriber_;
  const ros::Subscriber incremental_odometry_subscriber_;
  const ros::Publisher imu_incremental_odometry_publisher_;

  const gtsam::Pose3 imu_to_lidar;
  const gtsam::Pose3 lidar_to_imu;

  const boost::shared_ptr<gtsam::PreintegrationParams> params_;
  const gtsam::imuBias::ConstantBias prior_imu_bias_;

  const gtsam::Vector between_noise_bias_;
  const IMUExtrinsic imu_extrinsic_;

  gtsam::PreintegratedImuMeasurements integrator_;

  bool systemInitialized;

  std::deque<sensor_msgs::Imu> imu_queue;

  gtsam::Pose3 pose_;
  gtsam::Vector3 velocity_;
  gtsam::imuBias::ConstantBias bias_;

  bool bias_estimated_;
  double last_imu_time_ = -1;

  int key = 1;

  ImuIntegratorFactory integrator_factory;
  StatePrediction state_predition;

  IMUPreintegration()
  : imu_subscriber_(
      nh.subscribe<sensor_msgs::Imu>(
        imuTopic, 2000, &IMUPreintegration::estimateImuOdometry,
        this, ros::TransportHints().tcpNoDelay())),
    incremental_odometry_subscriber_(
      nh.subscribe<nav_msgs::Odometry>(
        "lio_sam/mapping/odometry_incremental", 5, &IMUPreintegration::estimateBias,
        this, ros::TransportHints().tcpNoDelay())),
    imu_incremental_odometry_publisher_(
      nh.advertise<geometry_msgs::TransformStamped>(imu_incremental_odometry_topic, 2000)),
    imu_to_lidar(gtsam::Pose3(gtsam::Rot3::identity(), -extTrans)),
    lidar_to_imu(gtsam::Pose3(gtsam::Rot3::identity(), extTrans)),
    params_(initialIntegrationParams(imuGravity, imuAccNoise, imuGyrNoise)),
    prior_imu_bias_(Vector6d::Zero()),
    between_noise_bias_(
      (Vector6d() <<
        imuAccBiasN, imuAccBiasN, imuAccBiasN,
        imuGyrBiasN, imuGyrBiasN, imuGyrBiasN).finished()),
    imu_extrinsic_(IMUExtrinsic(extRot, extQRPY)),
    integrator_(gtsam::PreintegratedImuMeasurements(params_, prior_imu_bias_)),
    systemInitialized(false),
    bias_(gtsam::imuBias::ConstantBias::identity()),
    bias_estimated_(false),
    integrator_factory(ImuIntegratorFactory(params_))
  {
  }

  void estimateBias(const nav_msgs::Odometry::ConstPtr & odom_msg)
  {
    std::lock_guard<std::mutex> lock(mtx);

    const double lidar_time = timeInSec(odom_msg->header);

    // make sure we have imu data to integrate
    if (!integrator_factory.isAvailable()) {
      return;
    }

    const gtsam::Pose3 lidar_pose = makeGtsamPose(odom_msg->pose.pose);
    const gtsam::Pose3 imu_pose = lidar_pose.compose(lidar_to_imu);

    if (!systemInitialized) {
      integrator_factory.init(lidar_time);

      state_predition = StatePrediction(imu_pose);
      key = 1;
      systemInitialized = true;
      return;
    }

    integrator_factory.build(bias_, lidar_time);
    const auto imu_integrator = integrator_factory.get();

    gtsam::NonlinearFactorGraph graph;

    const bool is_degenerate = odom_msg->pose.covariance[0] == 1;
    graph.add(makePrior(key, imu_pose, getCovariance(is_degenerate)));
    graph.add(makeImuConstraint(key, imu_integrator));
    graph.add(makeBiasConstraint(key, imu_integrator.deltaTij(), between_noise_bias_));

    std::tie(pose_, velocity_, bias_) = state_predition.update(key, imu_integrator, graph);

    if (!validateBias(bias_)) {
      ROS_WARN("Large bias, reset IMU-preintegration!");
      last_imu_time_ = -1;
      bias_estimated_ = false;
      systemInitialized = false;
      return;
    }

    if (!validateVelocity(velocity_)) {
      ROS_WARN("Large velocity, reset IMU-preintegration!");
      last_imu_time_ = -1;
      bias_estimated_ = false;
      systemInitialized = false;
      return;
    }

    ++key;
    bias_estimated_ = true;

    double last_imu_time = -1;

    popOldMessages(lidar_time, last_imu_time, imu_queue);

    const auto imus = imu_queue | ranges::to_vector;
    integrator_ = std::get<0>(makeIntegrator(params_, bias_, last_imu_time, imus));
  }

  void estimateImuOdometry(const sensor_msgs::Imu::ConstPtr & imu_raw)
  {
    std::lock_guard<std::mutex> lock(mtx);

    const sensor_msgs::Imu imu = extrinsicTransform(*imu_raw);
    integrator_factory.push(imu);
    imu_queue.push_back(imu);

    const double imu_time = timeInSec(imu.header);

    if (!bias_estimated_ || last_imu_time_ < 0) {
      last_imu_time_ = imu_time;
      return;
    }

    integrator_.integrateMeasurement(
      vector3ToEigen(imu.linear_acceleration),
      vector3ToEigen(imu.angular_velocity),
      imu_time - last_imu_time_
    );
    last_imu_time_ = imu_time;

    const auto current_imu = integrator_.predict(gtsam::NavState(pose_, velocity_), bias_);
    const auto lidar_pose = current_imu.pose().compose(imu_to_lidar);
    imu_incremental_odometry_publisher_.publish(
      makeTransformStamped(imu.header.stamp, odometryFrame, "odom_imu", makeTransform(lidar_pose))
    );
  }

  sensor_msgs::Imu extrinsicTransform(const sensor_msgs::Imu & imu_raw) const
  {
    try {
      return imu_extrinsic_.transform(imu_raw);
    } catch (const std::runtime_error & e) {
      ROS_ERROR(e.what());
      ros::shutdown();
      return sensor_msgs::Imu();
    }
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
