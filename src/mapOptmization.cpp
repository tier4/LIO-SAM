#include "utility.h"
#include "jacobian.h"
#include "homogeneous.h"
#include "param_server.h"
#include "pose_optimizer.hpp"
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
  const pcl::PointCloud<PointType> & laserCloudCornerLastDS,
  const pcl::PointCloud<PointType> & laserCloudSurfLastDS,
  const std::string & frame_id, const ros::Time & timestamp,
  const Vector6d & posevec)
{
  // publish registered key frame
  if (publisher.getNumSubscribers() == 0) {
    return;
  }

  pcl::PointCloud<PointType> cloudOut;
  cloudOut += transform(laserCloudCornerLastDS, posevec);
  cloudOut += transform(laserCloudSurfLastDS, posevec);
  sensor_msgs::PointCloud2 msg = toRosMsg(cloudOut);
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

bool checkPosesClose(
  const Vector6d & posevec0, const Vector6d & posevec1,
  const double angle_threshold, const double point_threshold)
{
  const Eigen::Affine3d affine0 = getTransformation(posevec0);
  const Eigen::Affine3d affine1 = getTransformation(posevec1);
  const auto [xyz, rpy] = getXYZRPY(affine0.inverse() * affine1);

  const bool f1 = (rpy.array() < angle_threshold).all();
  const bool f2 = xyz.norm() < point_threshold;
  return f1 && f2;
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

class mapOptimization : public ParamServer
{
  using CornerSurfaceDict = std::map<
    int,
    std::pair<pcl::PointCloud<PointType>,
    pcl::PointCloud<PointType>>
  >;

public:
  // gtsam
  const ros::Publisher pubLaserCloudSurround;
  const ros::Publisher pubLaserOdometryGlobal;
  const ros::Publisher pubLaserOdometryIncremental;
  const ros::Publisher pubKeyPoses;
  const ros::Publisher pubPath;

  const ros::Publisher pubRecentKeyFrames;
  const ros::Publisher pubRecentKeyFrame;
  const ros::Subscriber subCloud;
  const PoseOptimizer pose_optimizer_;

  Vector6d posevec;

  std::vector<pcl::PointCloud<PointType>> corner_cloud;
  std::vector<pcl::PointCloud<PointType>> surface_cloud;

  pcl::PointCloud<PointType>::Ptr points3d;
  pcl::PointCloud<StampedPose> poses6dof;

  CornerSurfaceDict corner_surface_dict;

  std::mutex mtx;

  std::vector<geometry_msgs::PoseStamped> path_poses_;

  Eigen::Affine3d lastImuTransformation;

  bool lastImuPreTransAvailable;
  Eigen::Affine3d lastImuPreTransformation;

  bool lastIncreOdomPubFlag;

  Eigen::Affine3d increOdomAffine; // incremental odometry in affine
  double last_time_sec;

  mapOptimization()
  : pubLaserCloudSurround(nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/map_global", 1)),
    pubLaserOdometryGlobal(nh.advertise<nav_msgs::Odometry>("lio_sam/mapping/odometry", 1)),
    pubLaserOdometryIncremental(
      nh.advertise<nav_msgs::Odometry>("lio_sam/mapping/odometry_incremental", 1)),
    pubKeyPoses(nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/trajectory", 1)),
    pubPath(nh.advertise<nav_msgs::Path>("lio_sam/mapping/path", 1)),
    pubRecentKeyFrames(nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/map_local", 1)),
    pubRecentKeyFrame(
      nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/cloud_registered", 1)),
    subCloud(nh.subscribe<lio_sam::cloud_info>(
        "lio_sam/feature/cloud_info", 1, &mapOptimization::laserCloudInfoHandler,
        this, ros::TransportHints().tcpNoDelay())),
    pose_optimizer_(PoseOptimizer(N_SCAN, Horizon_SCAN, numberOfCores)),
    posevec(Vector6d::Zero()),
    points3d(new pcl::PointCloud<PointType>()),
    lastImuPreTransAvailable(false),
    lastIncreOdomPubFlag(false),
    last_time_sec(-1.0)
  {
  }

  void laserCloudInfoHandler(const lio_sam::cloud_infoConstPtr & msgIn)
  {
    // extract time stamp
    const ros::Time timestamp = msgIn->header.stamp;

    // corner feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudCornerLast(new pcl::PointCloud<PointType>());

    // surf feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast(new pcl::PointCloud<PointType>());

    pcl::fromROSMsg(msgIn->cloud_corner, *laserCloudCornerLast);
    pcl::fromROSMsg(msgIn->cloud_surface, *laserCloudSurfLast);

    std::lock_guard<std::mutex> lock(mtx);

    if (timestamp.toSec() - last_time_sec < mappingProcessInterval) {
      return;
    }

    last_time_sec = timestamp.toSec();

    // save current transformation before any processing
    const Eigen::Affine3d front = getTransformation(posevec);

    updateInitialGuess(
      lastImuTransformation, msgIn->odomAvailable, msgIn->imuAvailable,
      msgIn->initialIMU, msgIn->initial_pose
    );

    lastImuTransformation = makeAffine(vector3ToEigen(msgIn->initialIMU));

    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMapDS(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS(new pcl::PointCloud<PointType>());

    extractSurroundingKeyFrames(
      timestamp, poses6dof,
      laserCloudCornerFromMapDS, laserCloudSurfFromMapDS, corner_surface_dict
    );

    pcl::VoxelGrid<PointType> corner_filter;
    corner_filter.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
    pcl::PointCloud<PointType> laserCloudCornerLastDS;
    corner_filter.setInputCloud(laserCloudCornerLast);
    corner_filter.filter(laserCloudCornerLastDS);

    pcl::VoxelGrid<PointType> surface_filter;
    surface_filter.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
    pcl::PointCloud<PointType> laserCloudSurfLastDS;
    surface_filter.setInputCloud(laserCloudSurfLast);
    surface_filter.filter(laserCloudSurfLastDS);

    bool isDegenerate = false;
    std::tie(posevec, isDegenerate) = scan2MapOptimization(
      laserCloudCornerLastDS, laserCloudSurfLastDS,
      laserCloudCornerFromMapDS, laserCloudSurfFromMapDS,
      msgIn->imuAvailable, msgIn->initialIMU, posevec
    );

    const Eigen::Affine3d back = getTransformation(posevec);
    const Eigen::Affine3d pose_increment = (front.inverse() * back);

    if (
      poses6dof.empty() ||
      !checkPosesClose(
        makePosevec(poses6dof.back()), posevec,
        surroundingkeyframeAddingAngleThreshold,
        surroundingkeyframeAddingDistThreshold))
    {
      // size can be used as index
      points3d->push_back(makePoint(posevec.tail(3), points3d->size()));

      poses6dof.push_back(makeStampedPose(posevec, timestamp.toSec()));

      // save key frame cloud
      corner_cloud.push_back(laserCloudCornerLastDS);
      surface_cloud.push_back(laserCloudSurfLastDS);

      path_poses_.push_back(makePoseStamped(makePose(posevec), odometryFrame, timestamp.toSec()));
    }

    const bool imuAvailable = msgIn->imuAvailable;
    const geometry_msgs::Vector3 initialIMU = msgIn->initialIMU;

    const nav_msgs::Odometry odometry = makeOdometry(
      timestamp, odometryFrame, "odom_mapping", makePose(posevec));

    pubLaserOdometryGlobal.publish(odometry);

    // Publish TF
    tf::TransformBroadcaster br;
    br.sendTransform(
      tf::StampedTransform(makeTransform(posevec), timestamp, odometryFrame, "lidar_link"));

    // Publish odometry for ROS (incremental)
    if (!lastIncreOdomPubFlag) {
      lastIncreOdomPubFlag = true;
      increOdomAffine = getTransformation(posevec);
      pubLaserOdometryIncremental.publish(odometry);
    } else {
      increOdomAffine = increOdomAffine * pose_increment;
      Vector6d incre_pose = getPoseVec(increOdomAffine);

      if (imuAvailable && std::abs(initialIMU.y) < 1.4) {
        const double imuWeight = 0.1;
        incre_pose(0) = interpolateRoll(incre_pose(0), initialIMU.x, imuWeight);
        incre_pose(1) = interpolatePitch(incre_pose(1), initialIMU.y, imuWeight);
      }

      nav_msgs::Odometry laserOdomIncremental = makeOdometry(
        timestamp, odometryFrame, "odom_mapping", makePose(incre_pose));
      if (isDegenerate) {
        laserOdomIncremental.pose.covariance[0] = 1;
      } else {
        laserOdomIncremental.pose.covariance[0] = 0;
      }
      pubLaserOdometryIncremental.publish(laserOdomIncremental);
    }

    if (points3d->empty()) {
      return;
    }

    // publish key poses
    publishCloud(pubKeyPoses, *points3d, timestamp, odometryFrame);
    // Publish surrounding key frames
    publishCloud(pubRecentKeyFrames, *laserCloudSurfFromMapDS, timestamp, odometryFrame);
    publishDownsampledCloud(
      pubRecentKeyFrame, laserCloudCornerLastDS, laserCloudSurfLastDS,
      odometryFrame, timestamp, posevec);
    publishPath(pubPath, odometryFrame, timestamp, path_poses_);
  }

  void updateInitialGuess(
    const Eigen::Affine3d & lastImuTransformation,
    const bool odomAvailable, const bool imuAvailable,
    const geometry_msgs::Vector3 & initialIMU,
    const geometry_msgs::Pose & initial_pose)
  {
    // initialization
    if (points3d->empty()) {
      const Eigen::Vector3d rpy = vector3ToEigen(initialIMU);

      posevec = initPosevec(rpy, useImuHeadingInitialization);

      // save imu before return;
      return;
    }

    // use imu pre-integration estimation for pose guess
    if (odomAvailable && lastImuPreTransAvailable) {
      const Eigen::Affine3d back = poseToAffine(initial_pose);
      const Eigen::Affine3d incre = lastImuPreTransformation.inverse() * back;

      lastImuPreTransformation = poseToAffine(initial_pose);

      const Eigen::Affine3d tobe = getTransformation(posevec);
      posevec = getPoseVec(tobe * incre);

      return;
    }

    if (odomAvailable) {
      lastImuPreTransformation = poseToAffine(initial_pose);
      lastImuPreTransAvailable = true;
    }

    // use imu incremental estimation for pose guess (only rotation)
    if (imuAvailable) {
      const Eigen::Vector3d rpy = vector3ToEigen(initialIMU);
      const Eigen::Affine3d back = makeAffine(rpy);
      const Eigen::Affine3d incre = lastImuTransformation.inverse() * back;
      const Eigen::Affine3d tobe = getTransformation(posevec);

      posevec = getPoseVec(tobe * incre);

      return;
    }
  }

  void extractSurroundingKeyFrames(
    const ros::Time & timestamp,
    const pcl::PointCloud<StampedPose> & poses6dof,
    const pcl::PointCloud<PointType>::Ptr & laserCloudCornerFromMapDS,
    const pcl::PointCloud<PointType>::Ptr & laserCloudSurfFromMapDS,
    CornerSurfaceDict & corner_surface_dict)
  {
    if (points3d->empty()) {
      return;
    }

    std::vector<int> indices;
    std::vector<float> squared_distances;
    pcl::KdTreeFLANN<PointType> kdtree;

    const double radius = (double)surroundingKeyframeSearchRadius;
    // extract all the nearby key poses and downsample them
    kdtree.setInputCloud(points3d); // create kd-tree
    kdtree.radiusSearch(points3d->back(), radius, indices, squared_distances);

    pcl::PointCloud<PointType>::Ptr poses(new pcl::PointCloud<PointType>());
    for (unsigned int index : indices) {
      poses->push_back(points3d->at(index));
    }

    pcl::PointCloud<PointType> downsampled = downsample(poses, surroundingKeyframeDensity);
    for (auto & pt : downsampled) {
      kdtree.nearestKSearch(pt, 1, indices, squared_distances);
      pt.intensity = points3d->at(indices[0]).intensity;
    }

    // also extract some latest key frames in case the robot rotates in one position
    for (int i = points3d->size() - 1; i >= 0; --i) {
      if (timestamp.toSec() - poses6dof.at(i).time >= 10.0) {
        break;
      }
      downsampled.push_back(points3d->at(i));
    }

    // fuse the map
    pcl::PointCloud<PointType>::Ptr corner(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr surface(new pcl::PointCloud<PointType>());

    for (auto & pt  : downsampled) {
      const double distance = (getXYZ(pt) - getXYZ(points3d->back())).norm();
      if (distance > radius) {
        continue;
      }

      const int index = static_cast<int>(pt.intensity);
      if (corner_surface_dict.find(index) != corner_surface_dict.end()) {
        // transformed cloud available
        *corner += corner_surface_dict[index].first;
        *surface += corner_surface_dict[index].second;
        continue;
      }

      // transformed cloud not available
      const Vector6d v = makePosevec(poses6dof.at(index));
      const pcl::PointCloud<PointType> c = transform(corner_cloud[index], v);
      const pcl::PointCloud<PointType> s = transform(surface_cloud[index], v);
      *corner += c;
      *surface += s;
      corner_surface_dict[index] = std::make_pair(c, s);
    }

    pcl::VoxelGrid<PointType> corner_filter;
    corner_filter.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
    corner_filter.setInputCloud(corner);
    corner_filter.filter(*laserCloudCornerFromMapDS);

    pcl::VoxelGrid<PointType> surface_filter;
    surface_filter.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
    surface_filter.setInputCloud(surface);
    surface_filter.filter(*laserCloudSurfFromMapDS);

    // clear map cache if too large
    if (corner_surface_dict.size() > 1000) {
      corner_surface_dict.clear();
    }
  }

  std::tuple<Vector6d, bool> scan2MapOptimization(
    const pcl::PointCloud<PointType> & laserCloudCornerLastDS,
    const pcl::PointCloud<PointType> & laserCloudSurfLastDS,
    const pcl::PointCloud<PointType>::Ptr & laserCloudCornerFromMapDS,
    const pcl::PointCloud<PointType>::Ptr & laserCloudSurfFromMapDS,
    const bool imuAvailable, const geometry_msgs::Vector3 & initialIMU,
    const Vector6d & initial_posevec) const
  {
    if (points3d->empty()) {
      return {initial_posevec, false};
    }

    if (
      edgeFeatureMinValidNum >= static_cast<int>(laserCloudCornerLastDS.size()) ||
      surfFeatureMinValidNum >= static_cast<int>(laserCloudSurfLastDS.size()))
    {
      ROS_WARN(
        "Not enough features! Only %d edge and %d planar features available.",
        laserCloudCornerLastDS.size(), laserCloudSurfLastDS.size());
      return {initial_posevec, false};
    }

    auto [posevec, isDegenerate] = pose_optimizer_.run(
      laserCloudCornerLastDS, laserCloudSurfLastDS,
      laserCloudCornerFromMapDS, laserCloudSurfFromMapDS, initial_posevec);

    if (imuAvailable && std::abs(initialIMU.y) < 1.4) {
      posevec(0) = interpolateRoll(posevec(0), initialIMU.x, imuRPYWeight);
      posevec(1) = interpolatePitch(posevec(1), initialIMU.y, imuRPYWeight);
    }

    posevec(0) = std::clamp(posevec(0), -rotation_tolerance, rotation_tolerance);
    posevec(1) = std::clamp(posevec(1), -rotation_tolerance, rotation_tolerance);
    posevec(5) = std::clamp(posevec(5), -z_tolerance, z_tolerance);
    return {posevec, isDegenerate};
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
