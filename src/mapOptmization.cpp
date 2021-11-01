#include "comprehend.hpp"
#include "message.hpp"
#include "downsample.hpp"
#include "utility.h"
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

using gtsam::symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using gtsam::symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using gtsam::symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)

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

pcl::PointCloud<PointType> transform(
  const pcl::PointCloud<PointType> & input, const Vector6d & posevec,
  const int numberOfCores = 2)
{
  pcl::PointCloud<PointType> output;

  output.resize(input.size());
  const Eigen::Affine3d transform = getTransformation(posevec);

  #pragma omp parallel for num_threads(numberOfCores)
  for (unsigned int i = 0; i < input.size(); ++i) {
    const auto & point = input.at(i);
    const Eigen::Vector3d p = getXYZ(point);
    output.at(i) = makePoint(transform * p, point.intensity);
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

void publishDownsampledCloud(
  const ros::Publisher & publisher,
  const pcl::PointCloud<PointType>::Ptr & corner,
  const pcl::PointCloud<PointType>::Ptr & surface,
  const std::string & frame_id, const ros::Time & timestamp,
  const Vector6d & posevec)
{
  // publish registered key frame
  if (publisher.getNumSubscribers() == 0) {
    return;
  }

  pcl::PointCloud<PointType> output;
  output += transform(*corner, posevec);
  output += transform(*surface, posevec);
  sensor_msgs::PointCloud2 msg = toRosMsg(output);
  msg.header.stamp = timestamp;
  msg.header.frame_id = frame_id;
  publisher.publish(msg);
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

void update3DPoints(
  const gtsam::Values & estimate,
  pcl::PointCloud<PointType>::Ptr & points3d)
{
  assert(points3d->size() == estimate.size());
  for (unsigned int i = 0; i < estimate.size(); ++i) {
    const gtsam::Point3 t = estimate.at<gtsam::Pose3>(i).translation();
    const float intensity = points3d->at(i).intensity;
    points3d->at(i) = makePoint(t, intensity);
  }
}

void update6DofPoses(
  const gtsam::Values & estimate,
  pcl::PointCloud<StampedPose> & poses6dof)
{
  assert(poses6dof.size() == estimate.size());
  for (unsigned int i = 0; i < estimate.size(); ++i) {
    const gtsam::Pose3 pose = estimate.at<gtsam::Pose3>(i);
    const double time = poses6dof.at(i).time;
    poses6dof.at(i) = makeStampedPose(pose, time);
  }
}

void updatePath(
  const gtsam::Values & estimate,
  const std::string & odometryFrame,
  const pcl::PointCloud<StampedPose> & poses6dof,
  std::vector<geometry_msgs::PoseStamped> path_poses)
{
  assert(path_poses.size() == estimate.size());
  for (unsigned int i = 0; i < estimate.size(); ++i) {
    const gtsam::Pose3 pose = estimate.at<gtsam::Pose3>(i);
    const double time = poses6dof.at(i).time;
    path_poses[i] = makePoseStamped(makePose(pose), odometryFrame, time);
  }
}

class MapFusion
{
public:
  MapFusion(
    const std::vector<pcl::PointCloud<PointType>::Ptr> & corner_cloud,
    const std::vector<pcl::PointCloud<PointType>::Ptr> & surface_cloud)
  : corner_cloud_(corner_cloud), surface_cloud_(surface_cloud)
  {
  }

  std::tuple<pcl::PointCloud<PointType>::Ptr, pcl::PointCloud<PointType>::Ptr>
  operator()(
    const pcl::PointCloud<PointType>::Ptr & points,
    const pcl::PointCloud<StampedPose> & poses6dof,
    const double radius) const
  {
    pcl::PointCloud<PointType>::Ptr corner(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr surface(new pcl::PointCloud<PointType>());

    const Eigen::Vector3d latest = getXYZ(poses6dof.back());

    for (auto & p : *points) {
      const double distance = (getXYZ(p) - latest).norm();
      if (distance > radius) {
        continue;
      }

      const int index = static_cast<int>(p.intensity);

      const Vector6d v = makePosevec(poses6dof.at(index));
      *corner += transform(*corner_cloud_[index], v);
      *surface += transform(*surface_cloud_[index], v);
    }

    return {corner, surface};
  }

private:
  const std::vector<pcl::PointCloud<PointType>::Ptr> & corner_cloud_;
  const std::vector<pcl::PointCloud<PointType>::Ptr> & surface_cloud_;
};

std::tuple<pcl::PointCloud<PointType>::Ptr, pcl::PointCloud<PointType>::Ptr>
extractSurroundingKeyFrames(
  const ros::Time & timestamp,
  const pcl::PointCloud<PointType>::Ptr & points3d,
  const pcl::PointCloud<StampedPose> & poses6dof,
  const MapFusion & map_fusion,
  const float radius,
  const float keyframe_density,
  const float corner_leaf_size,
  const float surface_leaf_size)
{
  const KDTree<PointType> kdtree(points3d);

  const auto r = kdtree.radiusSearch(points3d->back(), radius);
  const std::vector<int> indices = std::get<0>(r);

  auto points = downsample(comprehend(*points3d, indices), keyframe_density);
  for (auto & p : *points) {
    const int index = std::get<0>(kdtree.closestPoint(p));
    const auto closest = points3d->at(index);
    p.intensity = closest.intensity;
  }

  // also extract some latest key frames in case the robot rotates in one position
  for (int i = points3d->size() - 1; i >= 0; --i) {
    if (timestamp.toSec() - poses6dof.at(i).time >= 10.0) {
      break;
    }
    points->push_back(points3d->at(i));
  }

  const auto [corner, surface] = map_fusion(points, poses6dof, radius);
  return {
    downsample(corner, corner_leaf_size),
    downsample(surface, surface_leaf_size)
  };
}

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

  Vector6d posevec;

  std::vector<pcl::PointCloud<PointType>::Ptr> corner_cloud_;
  std::vector<pcl::PointCloud<PointType>::Ptr> surface_cloud_;

  pcl::PointCloud<PointType>::Ptr points3d;
  pcl::PointCloud<StampedPose> poses6dof;

  std::mutex mtx;

  std::vector<geometry_msgs::PoseStamped> path_poses_;

  std::optional<geometry_msgs::Vector3> last_imu_orientation;

  std::optional<geometry_msgs::Pose> last_imu_pose;

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
    posevec(Vector6d::Zero()),
    points3d(new pcl::PointCloud<PointType>()),
    last_imu_orientation(std::nullopt),
    last_imu_pose(std::nullopt),
    incremental_odometry(std::nullopt),
    last_time_sec(-1.0)
  {
  }

  void laserCloudInfoHandler(const lio_sam::cloud_infoConstPtr & msg)
  {
    // extract time stamp
    const ros::Time timestamp = msg->header.stamp;

    pcl::PointCloud<PointType>::Ptr corner_cloud = getPointCloud<PointType>(msg->cloud_corner);
    pcl::PointCloud<PointType>::Ptr surface_cloud = getPointCloud<PointType>(msg->cloud_surface);

    std::lock_guard<std::mutex> lock(mtx);

    if (timestamp.toSec() - last_time_sec < mappingProcessInterval) {
      return;
    }

    last_time_sec = timestamp.toSec();

    // save current transformation before any processing
    const Eigen::Affine3d front = getTransformation(posevec);

    updateInitialGuess(
      msg->imu_odometry_available, msg->imu_orientation_available,
      msg->imu_orientation, msg->scan_start_imu_pose
    );

    pcl::PointCloud<PointType>::Ptr corner = downsample(corner_cloud, mappingCornerLeafSize);
    pcl::PointCloud<PointType>::Ptr surface = downsample(surface_cloud, mappingSurfLeafSize);

    bool is_degenerate = false;
    if (!points3d->empty()) {
      const auto [corner_map, surface_map] = extractSurroundingKeyFrames(
        timestamp, points3d, poses6dof,
        MapFusion(corner_cloud_, surface_cloud_),
        surroundingKeyframeSearchRadius, surroundingKeyframeDensity,
        mappingCornerLeafSize, mappingSurfLeafSize
      );

      try {
        const CloudOptimizer cloud_optimizer(
          N_SCAN, Horizon_SCAN, numberOfCores,
          edgeFeatureMinValidNum, surfFeatureMinValidNum,
          corner, surface,
          corner_map, surface_map);

        std::tie(posevec, is_degenerate) = scan2MapOptimization(
          cloud_optimizer, msg->imu_orientation_available, msg->imu_orientation, posevec
        );
      } catch (const std::exception & e) {
        ROS_WARN(e.what());
      }
    }

    if (
      poses6dof.empty() ||
      isKeyframe(
        makePosevec(poses6dof.back()), posevec,
        keyframe_angle_threshold, keyframe_distance_threshold))
    {
      // size can be used as index
      points3d->push_back(makePoint(posevec.tail(3), points3d->size()));

      poses6dof.push_back(makeStampedPose(posevec, timestamp.toSec()));

      // save key frame cloud
      corner_cloud_.push_back(corner);
      surface_cloud_.push_back(surface);

      path_poses_.push_back(
        makePoseStamped(makePose(posevec), odometryFrame, timestamp.toSec()));
    }

    const nav_msgs::Odometry odometry = makeOdometry(
      timestamp, odometryFrame, "odom_mapping", makePose(posevec));

    pubLaserOdometryGlobal.publish(odometry);

    // Publish TF
    tf::TransformBroadcaster br;
    br.sendTransform(
      tf::StampedTransform(makeTransform(posevec), timestamp, odometryFrame, "lidar_link"));

    // Publish odometry for ROS (incremental)
    if (!incremental_odometry.has_value()) {
      incremental_odometry = getTransformation(posevec);
      pubLaserOdometryIncremental.publish(odometry);
    } else {
      const Eigen::Affine3d back = getTransformation(posevec);
      const Eigen::Affine3d pose_increment = (front.inverse() * back);

      incremental_odometry = incremental_odometry.value() * pose_increment;
      Vector6d pose = getPoseVec(incremental_odometry.value());

      if (msg->imu_orientation_available && std::abs(msg->imu_orientation.y) < 1.4) {
        const double weight = 0.1;
        pose(0) = interpolateRoll(pose(0), msg->imu_orientation.x, weight);
        pose(1) = interpolatePitch(pose(1), msg->imu_orientation.y, weight);
      }

      auto p = makeOdometry(timestamp, odometryFrame, "odom_mapping", makePose(pose));
      if (is_degenerate) {
        p.pose.covariance[0] = 1;
      } else {
        p.pose.covariance[0] = 0;
      }
      pubLaserOdometryIncremental.publish(p);
    }

    if (points3d->empty()) {
      return;
    }

    // publish key poses
    publishCloud(pubKeyPoses, *points3d, timestamp, odometryFrame);
    publishDownsampledCloud(
      pubRecentKeyFrame, corner, surface,
      odometryFrame, timestamp, posevec);
    publishPath(pubPath, odometryFrame, timestamp, path_poses_);
  }

  void updateInitialGuess(
    const bool imu_odometry_available, const bool imu_orientation_available,
    const geometry_msgs::Vector3 & imu_orientation,
    const geometry_msgs::Pose & scan_start_imu_pose)
  {
    // initialization
    if (points3d->empty()) {
      posevec = initPosevec(vector3ToEigen(imu_orientation), useImuHeadingInitialization);
      return;
    }

    // use imu pre-integration estimation for pose guess
    if (imu_odometry_available && last_imu_pose.has_value()) {
      const Eigen::Affine3d curr = poseToAffine(scan_start_imu_pose);
      const Eigen::Affine3d last = poseToAffine(last_imu_pose.value());
      const Eigen::Affine3d incre = last.inverse() * curr;

      last_imu_pose = scan_start_imu_pose;

      posevec = getPoseVec(getTransformation(posevec) * incre);
      return;
    }

    if (imu_odometry_available) {
      last_imu_pose = scan_start_imu_pose;
    }

    // use imu incremental estimation for pose guess (only rotation)
    if (imu_orientation_available && last_imu_orientation.has_value()) {
      const Eigen::Affine3d curr = makeAffine(vector3ToEigen(imu_orientation));
      const Eigen::Affine3d last = makeAffine(vector3ToEigen(last_imu_orientation.value()));
      const Eigen::Affine3d incre = last.inverse() * curr;

      last_imu_orientation = imu_orientation;

      posevec = getPoseVec(getTransformation(posevec) * incre);
      return;
    }

    if (imu_orientation_available) {
      last_imu_orientation = imu_orientation;
    }
  }

  std::tuple<Vector6d, bool> scan2MapOptimization(
    const CloudOptimizer & cloud_optimizer,
    const bool imu_orientation_available, const geometry_msgs::Vector3 & imu_orientation,
    const Vector6d & initial_posevec) const
  {
    assert(!points3d->empty());

    auto [posevec, is_degenerate] = optimizePose(cloud_optimizer, initial_posevec);

    if (imu_orientation_available && std::abs(imu_orientation.y) < 1.4) {
      posevec(0) = interpolateRoll(posevec(0), imu_orientation.x, imuRPYWeight);
      posevec(1) = interpolatePitch(posevec(1), imu_orientation.y, imuRPYWeight);
    }

    return {posevec, is_degenerate};
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
