#include "comprehend.hpp"
#include "message.hpp"
#include "downsample.hpp"
#include "utility.hpp"
#include "jacobian.h"
#include "homogeneous.h"
#include "param_server.h"
#include "pose_optimizer.hpp"
#include "kdtree.hpp"
#include "lio_sam/cloud_info.h"
#include "lio_sam/save_map.h"

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>

#include "range/v3/all.hpp"

using gtsam::symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using gtsam::symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using gtsam::symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)

class ImuOrientationIncrement
{
public:
  ImuOrientationIncrement()
  : last_pose_(std::nullopt), increment_(Eigen::Affine3d::Identity())
  {
  }

  bool is_available()
  {
    return last_pose_.has_value();
  }

  void init(const Eigen::Affine3d & pose)
  {
    last_pose_ = pose;
  }

  void compute(const Eigen::Affine3d & current_pose)
  {
    const Eigen::Affine3d last_pose = last_pose_.value();
    last_pose_ = current_pose;
    increment_ = last_pose.inverse() * current_pose;
  }

  Eigen::Affine3d get()
  {
    return increment_;
  }

private:
  std::optional<Eigen::Affine3d> last_pose_;
  Eigen::Affine3d increment_;
};

struct StampedPose
{
  PCL_ADD_POINT4D;  // preferred way of adding a XYZ
  float roll;
  float pitch;
  float yaw;
  double time;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT(
  StampedPose,
  (float, x, x)(float, y, y)(float, z, z)(float, roll, roll)(
    float, pitch, pitch)(float, yaw, yaw)(double, time, time)
)

StampedPose makeStampedPose(const Eigen::Vector3d rpy, const Eigen::Vector3d xyz, const double time)
{
  StampedPose pose6dof;
  pose6dof.x = xyz(0);
  pose6dof.y = xyz(1);
  pose6dof.z = xyz(2);
  pose6dof.roll = rpy(0);
  pose6dof.pitch = rpy(1);
  pose6dof.yaw = rpy(2);
  pose6dof.time = time;
  return pose6dof;
}

StampedPose makeStampedPose(const gtsam::Pose3 & pose, const double time)
{
  return makeStampedPose(pose.rotation().rpy(), pose.translation(), time);
}

StampedPose makeStampedPose(const Vector6d & posevec, const double time)
{
  return makeStampedPose(posevec.head(3), posevec.tail(3), time);
}

Eigen::Vector3d getXYZ(const StampedPose & pose)
{
  return Eigen::Vector3d(pose.x, pose.y, pose.z);
}

tf::Transform makeTransform(const Vector6d & posevec)
{
  return tf::Transform(
    tf::createQuaternionFromRPY(posevec(0), posevec(1), posevec(2)),
    tf::Vector3(posevec(3), posevec(4), posevec(5))
  );
}

Vector6d getPoseVec(const gtsam::Pose3 & pose)
{
  const gtsam::Rot3 r = pose.rotation();
  const gtsam::Point3 t = pose.translation();
  Vector6d v;
  v << r.roll(), r.pitch(), r.yaw(), t.x(), t.y(), t.z();
  return v;
}

Vector6d getPoseVec(const Eigen::Affine3d & transform)
{
  Vector6d posevec;
  pcl::getTranslationAndEulerAngles(
    transform,
    posevec(3), posevec(4), posevec(5),
    posevec(0), posevec(1), posevec(2));
  return posevec;
}

gtsam::Pose3 posevecToGtsamPose(const Vector6d & posevec)
{
  return gtsam::Pose3(
    gtsam::Rot3::RzRyRx(posevec(0), posevec(1), posevec(2)),
    gtsam::Point3(posevec(3), posevec(4), posevec(5)));
}

Vector6d makePosevec(const StampedPose & p)
{
  Vector6d v;
  v << p.roll, p.pitch, p.yaw, p.x, p.y, p.z;
  return v;
}

pcl::PointCloud<pcl::PointXYZ> transform(
  const Vector6d & posevec, const pcl::PointCloud<pcl::PointXYZ> & input,
  const int n_cores = 2)
{
  pcl::PointCloud<pcl::PointXYZ> output;

  output.resize(input.size());
  const Eigen::Affine3d affine = getTransformation(posevec);

  #pragma omp parallel for num_threads(n_cores)
  for (unsigned int i = 0; i < input.size(); ++i) {
    output.at(i) = transform(affine, input.at(i));
  }
  return output;
}

gtsam::PriorFactor<gtsam::Pose3> makePriorFactor(const Vector6d & posevec)
{
  // rad*rad, meter*meter
  const gtsam::Pose3 dst = posevecToGtsamPose(posevec);
  const Vector6d v = (Vector6d() << 1e-2, 1e-2, M_PI * M_PI, 1e8, 1e8, 1e8).finished();
  const auto noise = gtsam::noiseModel::Diagonal::Variances(v);
  return gtsam::PriorFactor<gtsam::Pose3>(0, dst, noise);
}

gtsam::BetweenFactor<gtsam::Pose3> makeOdomFactor(
  const Vector6d & last_pose,
  const Vector6d & posevec,
  const int size)
{
  const gtsam::Pose3 src = posevecToGtsamPose(last_pose);
  const gtsam::Pose3 dst = posevecToGtsamPose(posevec);

  const Vector6d v = (Vector6d() << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished();
  const auto noise = gtsam::noiseModel::Diagonal::Variances(v);

  return gtsam::BetweenFactor<gtsam::Pose3>(size - 1, size, src.between(dst), noise);
}

void publishPath(
  const ros::Publisher & publisher,
  const std::string & frame_id, const ros::Time & timestamp,
  const std::vector<geometry_msgs::PoseStamped> & poses)
{
  if (publisher.getNumSubscribers() == 0) {
    return;
  }
  nav_msgs::Path path;
  path.header.frame_id = frame_id;
  path.header.stamp = timestamp;
  path.poses = poses;
  publisher.publish(path);
}

Vector6d initPosevec(const Eigen::Vector3d & rpy, const bool useImuHeadingInitialization)
{
  Vector6d posevec = Vector6d::Zero();
  posevec.head(3) = rpy;

  if (!useImuHeadingInitialization) {
    posevec(2) = 0;
  }
  return posevec;
}

bool isKeyframe(
  const Vector6d & posevec0, const Vector6d & posevec1,
  const double angle_threshold, const double point_threshold)
{
  const Eigen::Affine3d affine0 = getTransformation(posevec0);
  const Eigen::Affine3d affine1 = getTransformation(posevec1);
  const auto [xyz, rpy] = getXYZRPY(affine0.inverse() * affine1);

  const bool f1 = (rpy.array() < angle_threshold).all();
  const bool f2 = xyz.norm() < point_threshold;
  return !(f1 && f2);
}

double interpolateRoll(const double r0, const double r1, const double weight)
{
  return interpolate(Eigen::Vector3d(r0, 0, 0), Eigen::Vector3d(r1, 0, 0), weight)(0);
}

double interpolatePitch(const double p0, const double p1, const double weight)
{
  return interpolate(Eigen::Vector3d(0, p0, 0), Eigen::Vector3d(0, p1, 0), weight)(1);
}

std::vector<int> findClosePositions(
  const pcl::PointCloud<StampedPose> & poses6dof,
  const std::vector<int> & input_indices,
  const StampedPose & query,
  const double radius)
{
  const Eigen::Vector3d p = getXYZ(query);
  const auto f = [&](int i) {return (getXYZ(poses6dof.at(i)) - p).norm() <= radius;};
  return input_indices | ranges::views::filter(f) | ranges::to_vector;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr mapFusion(
  const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> & cloud,
  const pcl::PointCloud<StampedPose> & poses6dof,
  const std::vector<int> & indices)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr fused(new pcl::PointCloud<pcl::PointXYZ>());

  for (const int index : indices) {
    const Vector6d v = makePosevec(poses6dof.at(index));
    *fused += transform(v, *cloud[index]);
  }

  return fused;
}

auto recentKeyframes(const std::vector<double> & timestamps, const double & current_timestamp)
{
  const auto is_recent = [&](int i) {return current_timestamp - timestamps.at(i) < 10.0;};
  const int n = timestamps.size();
  return ranges::views::iota(0, n) | ranges::views::filter(is_recent);
}

class KeyframeExtraction
{
public:
  KeyframeExtraction(const float radius, const float keyframe_density)
  : radius_(radius), keyframe_density_(keyframe_density)
  {
  }

  std::vector<int> operator()(
    const double & current_timestamp,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr & positions,
    const std::vector<double> & timestamps) const
  {
    const KDTree<pcl::PointXYZ> kdtree(positions);
    const auto r = kdtree.radiusSearch(positions->back(), radius_);
    const std::vector<int> indices = std::get<0>(r);
    const auto close_positions = comprehend(*positions, indices);
    const auto points = downsample<pcl::PointXYZ>(close_positions, keyframe_density_);
    auto f = [&kdtree](const pcl::PointXYZ & p) {return std::get<0>(kdtree.closestPoint(p));};
    auto surrounding = *points | ranges::views::transform(f);

    auto recent = recentKeyframes(timestamps, current_timestamp);
    auto merged = ranges::views::concat(surrounding, recent);
    return merged | ranges::to_vector;
  }

private:
  const float radius_;
  const float keyframe_density_;
};

class mapOptimization : public ParamServer
{
public:
  // gtsam
  const ros::Publisher pubLaserCloudSurround;
  const ros::Publisher pubLaserOdometryGlobal;
  const ros::Publisher pubLaserOdometryIncremental;
  const ros::Publisher pubKeyPoses;
  const ros::Publisher pubPath;

  const ros::Publisher pubRecentKeyFrame;
  const ros::Subscriber subCloud;
  const KeyframeExtraction extract_keyframes_;

  Vector6d posevec;

  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> edge_cloud_;
  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> surface_cloud_;

  pcl::PointCloud<pcl::PointXYZ>::Ptr positions;
  pcl::PointCloud<StampedPose> poses6dof_;

  std::mutex mtx;

  std::vector<double> timestamps_sec_;
  std::vector<geometry_msgs::PoseStamped> path_poses_;

  ImuOrientationIncrement imu_orientation_increment_;
  ImuOrientationIncrement imu_pose_increment_;

  std::optional<Eigen::Affine3d> incremental_odometry;  // incremental odometry in affine
  double last_time_sec;

  mapOptimization()
  : pubLaserCloudSurround(nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/map_global", 1)),
    pubLaserOdometryGlobal(nh.advertise<nav_msgs::Odometry>("lio_sam/mapping/odometry", 1)),
    pubLaserOdometryIncremental(
      nh.advertise<nav_msgs::Odometry>("lio_sam/mapping/odometry_incremental", 1)),
    pubKeyPoses(nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/trajectory", 1)),
    pubPath(nh.advertise<nav_msgs::Path>("lio_sam/mapping/path", 1)),
    pubRecentKeyFrame(
      nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/cloud_registered", 1)),
    subCloud(nh.subscribe<lio_sam::cloud_info>(
        "lio_sam/feature/cloud_info", 1, &mapOptimization::laserCloudInfoHandler,
        this, ros::TransportHints().tcpNoDelay())),
    extract_keyframes_(KeyframeExtraction(keyframe_search_radius, keyframe_density)),
    posevec(Vector6d::Zero()),
    positions(new pcl::PointCloud<pcl::PointXYZ>()),
    incremental_odometry(std::nullopt),
    last_time_sec(-1.0)
  {
  }

  void laserCloudInfoHandler(const lio_sam::cloud_infoConstPtr & msg)
  {
    // extract time stamp
    const ros::Time timestamp = msg->header.stamp;

    std::lock_guard<std::mutex> lock(mtx);

    const double curr_time_sec = timestamp.toSec();
    if (curr_time_sec - last_time_sec < map_process_interval) {
      return;
    }

    last_time_sec = curr_time_sec;

    // save current transformation before any processing
    const Eigen::Affine3d front = getTransformation(posevec);

    updateInitialGuess(
      msg->imu_odometry_available, msg->imu_orientation_available,
      msg->imu_orientation, msg->scan_start_imu_pose
    );

    const auto edge = getPointCloud<pcl::PointXYZ>(msg->cloud_edge);
    const auto surface = getPointCloud<pcl::PointXYZ>(msg->cloud_surface);

    bool is_degenerate = false;
    if (!positions->empty()) {
      const auto keyframe_indices = extract_keyframes_(curr_time_sec, positions, timestamps_sec_);
      const float radius = keyframe_search_radius;
      const auto latest = poses6dof_.back();
      const auto indices = findClosePositions(poses6dof_, keyframe_indices, latest, radius);
      const auto edge_fused = mapFusion(edge_cloud_, poses6dof_, indices);
      const auto surface_fused = mapFusion(surface_cloud_, poses6dof_, indices);
      const auto edge_map = downsample<pcl::PointXYZ>(edge_fused, map_edge_leaf_size);
      const auto surface_map = downsample<pcl::PointXYZ>(surface_fused, map_surface_leaf_size);

      const CloudOptimizer cloud_optimizer(n_cores, edge, surface, edge_map, surface_map);
      const bool is_degenerate = isDegenerate(cloud_optimizer, posevec);

      if (
        (static_cast<int>(edge->size()) > min_edge_cloud ||
        static_cast<int>(surface->size()) > min_surface_cloud) &&
        !is_degenerate)
      {
        posevec = optimizePose(cloud_optimizer, posevec);
      } else {
        ROS_WARN(
          "Not enough features! Only %d edge and %d planar features available.",
          edge->size(), surface->size()
        );
      }
    }

    if (
      poses6dof_.empty() ||
      isKeyframe(
        makePosevec(poses6dof_.back()), posevec,
        keyframe_angle_threshold, keyframe_distance_threshold))
    {
      // size can be used as index
      timestamps_sec_.push_back(curr_time_sec);
      positions->push_back(makePointXYZ(posevec.tail(3)));
      poses6dof_.push_back(makeStampedPose(posevec, curr_time_sec));

      // save key frame cloud
      edge_cloud_.push_back(edge);
      surface_cloud_.push_back(surface);

      path_poses_.push_back(makePoseStamped(makePose(posevec), odometryFrame, curr_time_sec));
    }

    const nav_msgs::Odometry odometry = makeOdometry(
      timestamp, odometryFrame, "odom_mapping", makePose(posevec));

    pubLaserOdometryGlobal.publish(odometry);

    // Publish odometry for ROS (incremental)
    if (!incremental_odometry.has_value()) {
      incremental_odometry = getTransformation(posevec);
      pubLaserOdometryIncremental.publish(odometry);
    } else {
      const Eigen::Affine3d back = getTransformation(posevec);
      const Eigen::Affine3d increment = (front.inverse() * back);

      incremental_odometry = incremental_odometry.value() * increment;
      Vector6d pose = getPoseVec(incremental_odometry.value());

      auto p = makeOdometry(timestamp, odometryFrame, "odom_mapping", makePose(pose));
      if (is_degenerate) {
        p.pose.covariance[0] = 1;
      } else {
        p.pose.covariance[0] = 0;
      }
      pubLaserOdometryIncremental.publish(p);
    }

    if (positions->empty()) {
      return;
    }

    // publish key poses
    pubKeyPoses.publish(toRosMsg(*positions, timestamp, odometryFrame));
    const auto output = transform(posevec, *edge) + transform(posevec, *surface);
    pubRecentKeyFrame.publish(toRosMsg(output, timestamp, odometryFrame));
    publishPath(pubPath, odometryFrame, timestamp, path_poses_);
  }

  void updateInitialGuess(
    const bool imu_odometry_available, const bool imu_orientation_available,
    const geometry_msgs::Vector3 & imu_orientation,
    const geometry_msgs::Pose & scan_start_imu_pose)
  {
    // initialization
    if (positions->empty()) {
      posevec = initPosevec(vector3ToEigen(imu_orientation), useImuHeadingInitialization);
      return;
    }

    // use imu pre-integration estimation for pose guess
    if (imu_odometry_available && imu_pose_increment_.is_available()) {
      imu_pose_increment_.compute(poseToAffine(scan_start_imu_pose));
      const Eigen::Affine3d increment = imu_pose_increment_.get();
      posevec = getPoseVec(getTransformation(posevec) * increment);
      return;
    }

    if (imu_odometry_available) {
      imu_pose_increment_.init(poseToAffine(scan_start_imu_pose));
    }

    if (imu_orientation_available && imu_orientation_increment_.is_available()) {
      imu_orientation_increment_.compute(makeAffine(vector3ToEigen(imu_orientation)));
      const Eigen::Affine3d increment = imu_orientation_increment_.get();
      posevec = getPoseVec(getTransformation(posevec) * increment);
      return;
    }

    if (imu_orientation_available) {
      imu_orientation_increment_.init(makeAffine(vector3ToEigen(imu_orientation)));
    }
  }
};

int main(int argc, char ** argv)
{
  ros::init(argc, argv, "lio_sam");

  mapOptimization MO;

  ROS_INFO("\033[1;32m----> Map Optimization Started.\033[0m");

  ros::spin();

  return 0;
}
