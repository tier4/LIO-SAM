#include "utility.h"
#include "jacobian.h"
#include "homogeneous.h"
#include "param_server.h"
#include "lio_sam/cloud_info.h"
#include "lio_sam/save_map.h"

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

using gtsam::symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using gtsam::symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using gtsam::symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)
using gtsam::symbol_shorthand::G; // GPS pose

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

StampedPose makeStampedPose(const gtsam::Pose3 & pose, const double time)
{
  const Eigen::Vector3d xyz = pose.translation();
  const Eigen::Vector3d rpy = pose.rotation().rpy();

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

tf::Quaternion tfQuaternionFromRPY(const Eigen::Vector3d & rpy)
{
  tf::Quaternion q;
  q.setRPY(rpy(0), rpy(1), rpy(2));
  return q;
}

Eigen::Vector3d getRPY(const tf::Quaternion & q)
{
  double roll, pitch, yaw;
  tf::Matrix3x3(q).getRPY(roll, pitch, yaw);
  return Eigen::Vector3d(roll, pitch, yaw);
}

tf::Quaternion interpolate(
  const tf::Quaternion & q0, const tf::Quaternion & q1,
  const tfScalar weight)
{
  return q0.slerp(q1, weight);
}

Eigen::Vector3d interpolate(
  const Eigen::Vector3d & rpy0, const Eigen::Vector3d & rpy1, const tfScalar weight)
{
  const tf::Quaternion q0 = tfQuaternionFromRPY(rpy0);
  const tf::Quaternion q1 = tfQuaternionFromRPY(rpy1);
  return getRPY(interpolate(q0, q1, weight));
}

float constraintTransformation(const float value, const float limit)
{
  if (value < -limit) {
    return -limit;
  }
  if (value > limit) {
    return limit;
  }

  return value;
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

bool validatePlane(
  const Eigen::Matrix<double, 5, 3> & A,
  const Eigen::Vector3d & x)
{
  const Eigen::Vector4d y = toHomogeneous(x) / x.norm();

  for (int j = 0; j < 5; j++) {
    const Eigen::Vector3d p = A.row(j);
    const Eigen::Vector4d q = toHomogeneous(p);

    if (fabs(y.transpose() * q) > 0.2) {
      return false;
    }
  }
  return true;
}

Eigen::Matrix<double, 5, 3> makeMatrixA(
  const pcl::PointCloud<PointType>::Ptr & pointcloud,
  const std::vector<int> & indices)
{
  Eigen::Matrix<double, 5, 3> A = Eigen::Matrix<double, 5, 3>::Zero();
  for (int j = 0; j < 5; j++) {
    A.row(j) = getXYZ(pointcloud->at(indices[j]));
  }
  return A;
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
  const pcl::PointCloud<StampedPose> & poses6dof, const Vector6d & posevec)
{
  const gtsam::Pose3 src = posevecToGtsamPose(makePosevec(poses6dof.points.back()));
  const gtsam::Pose3 dst = posevecToGtsamPose(posevec);

  const Vector6d v = (Vector6d() << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished();
  const auto noise = gtsam::noiseModel::Diagonal::Variances(v);

  const unsigned int size = poses6dof.size();

  return gtsam::BetweenFactor<gtsam::Pose3>(size - 1, size, src.between(dst), noise);
}

std::optional<gtsam::GPSFactor> makeGPSFactor(
  const pcl::PointCloud<PointType>::Ptr & points3d,
  const float gpsCovThreshold, const bool useGpsElevation,
  const Vector6d & posevec, const Eigen::Vector3d & last_gps_position,
  const ros::Time & timestamp,
  std::deque<nav_msgs::Odometry> & gpsQueue)
{

  if (gpsQueue.empty()) {
    return std::nullopt;
  }

  const double distance = (getXYZ(points3d->front()) - getXYZ(points3d->back())).norm();
  if (distance < 5.0) {
    return std::nullopt;
  }

  dropBefore(timestamp.toSec() - 0.2, gpsQueue);

  if (timeInSec(gpsQueue.front().header) > timestamp.toSec() + 0.2) {
    // message too new
    return std::nullopt;
  }

  while (!gpsQueue.empty()) {
    const geometry_msgs::PoseWithCovariance pose = gpsQueue.front().pose;
    gpsQueue.pop_front();

    // GPS too noisy, skip
    const Eigen::Map<const RowMajorMatrixXd> covariance(pose.covariance.data(), 6, 6);
    Eigen::Vector3d position_variances = covariance.diagonal().head(3);

    if (position_variances(0) > gpsCovThreshold || position_variances(1) > gpsCovThreshold) {
      continue;
    }

    Eigen::Vector3d gps_position = pointToEigen(pose.pose.position);
    if (!useGpsElevation) {
      gps_position(2) = posevec(5);
      position_variances(2) = 0.01;
    }

    // GPS not properly initialized (0,0,0)
    if (abs(gps_position(0)) < 1e-6 && abs(gps_position(1)) < 1e-6) {
      continue;
    }

    // Add GPS every a few meters
    if ((gps_position - last_gps_position).norm() < 5.0) {
      continue;
    }

    const auto gps_noise = gtsam::noiseModel::Diagonal::Variances(
      position_variances.cwiseMax(1.0f)
    );
    return std::make_optional<gtsam::GPSFactor>(points3d->size(), gps_position, gps_noise);
  }

  return std::nullopt;
}

class GPSFactor
{
public:
  void handler(const nav_msgs::Odometry::ConstPtr & gpsMsg)
  {
    gpsQueue.push_back(*gpsMsg);
  }

  std::optional<gtsam::GPSFactor> make(
    const pcl::PointCloud<PointType>::Ptr & points3d,
    const float gpsCovThreshold, const bool useGpsElevation,
    const Vector6d & posevec, const Eigen::Vector3d & last_gps_position,
    const ros::Time & timestamp)
  {
    return makeGPSFactor(
      points3d, gpsCovThreshold, useGpsElevation,
      posevec, last_gps_position, timestamp, gpsQueue
    );
  }

private:
  std::deque<nav_msgs::Odometry> gpsQueue;
};

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

class mapOptimization : public ParamServer
{
  using CornerSurfaceDict = std::map<
    int,
    std::pair<pcl::PointCloud<PointType>,
    pcl::PointCloud<PointType>>
  >;

public:
  // gtsam
  Eigen::MatrixXd poseCovariance;

  const ros::Publisher pubLaserCloudSurround;
  const ros::Publisher pubLaserOdometryGlobal;
  const ros::Publisher pubLaserOdometryIncremental;
  const ros::Publisher pubKeyPoses;
  const ros::Publisher pubPath;

  const ros::Publisher pubRecentKeyFrames;
  const ros::Publisher pubRecentKeyFrame;
  const ros::Subscriber subCloud;
  const ros::Subscriber subGPS;

  Vector6d posevec;

  std::shared_ptr<gtsam::ISAM2> isam;

  GPSFactor gps_factor_;
  lio_sam::cloud_infoConstPtr msgIn_;

  std::vector<pcl::PointCloud<PointType>> corner_cloud;
  std::vector<pcl::PointCloud<PointType>> surface_cloud;

  pcl::PointCloud<PointType>::Ptr points3d;
  pcl::PointCloud<StampedPose> poses6dof;

  CornerSurfaceDict corner_surface_dict;

  std::mutex mtx;

  bool aLoopIsClosed;

  std::vector<geometry_msgs::PoseStamped> path_poses_;

  Eigen::Affine3d incrementalOdometryAffineBack;
  Eigen::Affine3d lastImuTransformation;
  Eigen::Vector3d last_gps_position;

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
    subGPS(nh.subscribe<nav_msgs::Odometry>(
        gpsTopic, 200, &GPSFactor::handler, &gps_factor_,
        ros::TransportHints().tcpNoDelay())),
    posevec(Vector6d::Zero()),
    isam(std::make_shared<gtsam::ISAM2>(
        gtsam::ISAM2Params(gtsam::ISAM2GaussNewtonParams(), 0.1, 1))),
    points3d(new pcl::PointCloud<PointType>()),
    aLoopIsClosed(false),
    lastImuPreTransAvailable(false),
    lastIncreOdomPubFlag(false),
    last_time_sec(-1.0)
  {
  }

  void laserCloudInfoHandler(const lio_sam::cloud_infoConstPtr & msgIn)
  {
    msgIn_ = msgIn;

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
    const Vector6d front_posevec = posevec;
    updateInitialGuess(
      lastImuTransformation, msgIn->odomAvailable, msgIn->imuAvailable,
      msgIn->initialIMU, msgIn->initial_pose
    );

    lastImuTransformation = makeAffine(vector3ToEigen(msgIn_->initialIMU));

    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMapDS(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS(new pcl::PointCloud<PointType>());

    extractSurroundingKeyFrames(
      timestamp, poses6dof,
      laserCloudCornerFromMapDS, laserCloudSurfFromMapDS, corner_surface_dict
    );

    pcl::VoxelGrid<PointType> downSizeFilterCorner;
    downSizeFilterCorner.setLeafSize(
      mappingCornerLeafSize,
      mappingCornerLeafSize,
      mappingCornerLeafSize);
    pcl::PointCloud<PointType> laserCloudCornerLastDS;
    downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
    downSizeFilterCorner.filter(laserCloudCornerLastDS);

    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
    pcl::PointCloud<PointType> laserCloudSurfLastDS;
    downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
    downSizeFilterSurf.filter(laserCloudSurfLastDS);

    bool isDegenerate = false;
    scan2MapOptimization(
      laserCloudCornerLastDS, laserCloudSurfLastDS,
      laserCloudCornerFromMapDS, laserCloudSurfFromMapDS, isDegenerate, posevec
    );

    incrementalOdometryAffineBack = getTransformation(posevec);

    saveKeyFramesAndFactor(
      timestamp, laserCloudCornerLastDS, laserCloudSurfLastDS,
      corner_surface_dict
    );

    publishOdometry(
      timestamp, front_posevec, isDegenerate, posevec,
      lastIncreOdomPubFlag, increOdomAffine);

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

    // Downsample the surrounding corner key frames (or map)
    pcl::VoxelGrid<PointType> downSizeFilterCorner;
    downSizeFilterCorner.setLeafSize(
      mappingCornerLeafSize,
      mappingCornerLeafSize,
      mappingCornerLeafSize);
    downSizeFilterCorner.setInputCloud(corner);
    downSizeFilterCorner.filter(*laserCloudCornerFromMapDS);
    // Downsample the surrounding surf key frames (or map)
    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    downSizeFilterSurf.setLeafSize(
      mappingSurfLeafSize,
      mappingSurfLeafSize,
      mappingSurfLeafSize);
    downSizeFilterSurf.setInputCloud(surface);
    downSizeFilterSurf.filter(*laserCloudSurfFromMapDS);

    // clear map cache if too large
    if (corner_surface_dict.size() > 1000) {
      corner_surface_dict.clear();
    }
  }

  std::tuple<pcl::PointCloud<PointType>, pcl::PointCloud<PointType>>
  optimization(
    const pcl::PointCloud<PointType> & laserCloudCornerLastDS,
    const pcl::PointCloud<PointType> & laserCloudSurfLastDS,
    const pcl::KdTreeFLANN<PointType> & kdtreeCornerFromMap,
    const pcl::KdTreeFLANN<PointType> & kdtreeSurfFromMap,
    const pcl::PointCloud<PointType>::Ptr & laserCloudCornerFromMapDS,
    const pcl::PointCloud<PointType>::Ptr & laserCloudSurfFromMapDS) const
  {
    const Eigen::Affine3d point_to_map = getTransformation(posevec);
    std::vector<PointType> laserCloudOriCornerVec(N_SCAN * Horizon_SCAN);
    std::vector<PointType> coeffSelCornerVec(N_SCAN * Horizon_SCAN);
    // corner point holder for parallel computation
    std::vector<bool> laserCloudOriCornerFlag(N_SCAN * Horizon_SCAN, false);

    // corner optimization
    #pragma omp parallel for num_threads(numberOfCores)
    for (unsigned int i = 0; i < laserCloudCornerLastDS.size(); i++) {
      std::vector<int> indices;
      std::vector<float> squared_distances;

      const PointType point = laserCloudCornerLastDS.at(i);
      const Eigen::Vector3d map_point = point_to_map * getXYZ(point);
      const PointType p = makePoint(map_point, point.intensity);
      kdtreeCornerFromMap.nearestKSearch(p, 5, indices, squared_distances);

      if (squared_distances[4] < 1.0) {
        Eigen::Vector3d c = Eigen::Vector3d::Zero();
        for (int j = 0; j < 5; j++) {
          c += getXYZ(laserCloudCornerFromMapDS->at(indices[j]));
        }
        c /= 5.0;

        Eigen::Matrix3d sa = Eigen::Matrix3d::Zero();

        for (int j = 0; j < 5; j++) {
          const Eigen::Vector3d x = getXYZ(laserCloudCornerFromMapDS->at(indices[j]));
          const Eigen::Vector3d a = x - c;
          sa += a * a.transpose();
        }

        sa = sa / 5.0;

        const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(sa);
        const Eigen::Vector3d d1 = solver.eigenvalues();
        const Eigen::Matrix3d v1 = solver.eigenvectors();

        if (d1(0) > 3 * d1(1)) {
          const Eigen::Vector3d p0 = map_point;
          const Eigen::Vector3d p1 = c + 0.1 * v1.row(0).transpose();
          const Eigen::Vector3d p2 = c - 0.1 * v1.row(0).transpose();

          const Eigen::Vector3d d01 = p0 - p1;
          const Eigen::Vector3d d02 = p0 - p2;
          const Eigen::Vector3d d12 = p1 - p2;

          // const Eigen::Vector3d d012(d01(0) * d02(1) - d02(0) * d01(1),
          //                            d01(0) * d02(2) - d02(0) * d01(2),
          //                            d01(1) * d02(2) - d02(1) * d01(2));
          const Eigen::Vector3d cross(
            d01(1) * d02(2) - d01(2) * d02(1),
            d01(2) * d02(0) - d01(0) * d02(2),
            d01(0) * d02(1) - d01(1) * d02(0));

          const double a012 = cross.norm();

          const double l12 = d12.norm();

          // possible bag. maybe the commented one is correct
          // const Eigen::Vector3d v(
          //   (d12(1) * cross(2) - cross(2) * d12(1)),
          //   (d12(2) * cross(0) - cross(0) * d12(2)),
          //   (d12(0) * cross(1) - cross(1) * d12(0)));

          const Eigen::Vector3d v(
            (d12(1) * cross(2) - d12(2) * cross(1)),
            (d12(2) * cross(0) - d12(0) * cross(2)),
            (d12(0) * cross(1) - d12(1) * cross(0)));

          const double ld2 = a012 / l12;

          const double s = 1 - 0.9 * fabs(ld2);

          if (s > 0.1) {
            laserCloudOriCornerVec[i] = point;
            coeffSelCornerVec[i] = makePoint(s * v / (a012 * l12), s * ld2);
            laserCloudOriCornerFlag[i] = true;
          }
        }
      }
    }

    std::vector<PointType> laserCloudOriSurfVec(N_SCAN * Horizon_SCAN);
    std::vector<PointType> coeffSelSurfVec(N_SCAN * Horizon_SCAN);

    // surf point holder for parallel computation
    std::vector<bool> laserCloudOriSurfFlag(N_SCAN * Horizon_SCAN, false);

    // surface optimization
    #pragma omp parallel for num_threads(numberOfCores)
    for (unsigned int i = 0; i < laserCloudSurfLastDS.size(); i++) {
      std::vector<int> indices;
      std::vector<float> squared_distances;

      const PointType point = laserCloudSurfLastDS.at(i);
      const Eigen::Vector3d map_point = point_to_map * getXYZ(point);
      const PointType p = makePoint(map_point, point.intensity);
      kdtreeSurfFromMap.nearestKSearch(p, 5, indices, squared_distances);

      if (squared_distances[4] >= 1.0) {
        continue;
      }

      const Eigen::Matrix<double, 5, 1> b = -1.0 * Eigen::Matrix<double, 5, 1>::Ones();
      const Eigen::Matrix<double, 5, 3> A = makeMatrixA(laserCloudSurfFromMapDS, indices);
      const Eigen::Vector3d x = A.colPivHouseholderQr().solve(b);

      if (!validatePlane(A, x)) {
        continue;
      }

      const Eigen::Vector4d y = toHomogeneous(x) / x.norm();
      const Eigen::Vector4d q = toHomogeneous(map_point);
      const float pd2 = y.transpose() * q;
      const float s = 1 - 0.9 * fabs(pd2) / sqrt(map_point.norm());

      if (s <= 0.1) {
        continue;
      }

      laserCloudOriSurfVec[i] = point;
      coeffSelSurfVec[i] = makePoint((s / x.norm()) * x, s * pd2);
      laserCloudOriSurfFlag[i] = true;
    }

    pcl::PointCloud<PointType> laserCloudOri;
    pcl::PointCloud<PointType> coeffSel;

    // combine corner coeffs
    for (unsigned int i = 0; i < laserCloudCornerLastDS.size(); ++i) {
      if (laserCloudOriCornerFlag[i]) {
        laserCloudOri.push_back(laserCloudOriCornerVec[i]);
        coeffSel.push_back(coeffSelCornerVec[i]);
      }
    }
    // combine surf coeffs
    for (unsigned int i = 0; i < laserCloudSurfLastDS.size(); ++i) {
      if (laserCloudOriSurfFlag[i]) {
        laserCloudOri.push_back(laserCloudOriSurfVec[i]);
        coeffSel.push_back(coeffSelSurfVec[i]);
      }
    }

    return {laserCloudOri, coeffSel};
  }

  bool LMOptimization(
    const pcl::PointCloud<PointType> & laserCloudOri,
    const pcl::PointCloud<PointType> & coeffSel,
    const int iterCount, bool & isDegenerate, Vector6d & posevec) const
  {
    // This optimization is from the original loam_velodyne by Ji Zhang,
    // need to cope with coordinate transformation
    // lidar <- camera      ---     camera <- lidar
    // x = z                ---     x = y
    // y = x                ---     y = z
    // z = y                ---     z = x
    // roll = yaw           ---     roll = pitch
    // pitch = roll         ---     pitch = yaw
    // yaw = pitch          ---     yaw = roll

    // lidar -> camera
    int laserCloudSelNum = laserCloudOri.size();
    if (laserCloudSelNum < 50) {
      return false;
    }

    Eigen::MatrixXd A(laserCloudSelNum, 6);
    Eigen::VectorXd b(laserCloudSelNum);

    for (int i = 0; i < laserCloudSelNum; i++) {
      // lidar -> camera
      const float intensity = coeffSel.at(i).intensity;

      // in camera

      const Eigen::Vector3d point_ori(
        laserCloudOri.at(i).y,
        laserCloudOri.at(i).z,
        laserCloudOri.at(i).x);

      const Eigen::Vector3d coeff_vec(
        coeffSel.at(i).y,
        coeffSel.at(i).z,
        coeffSel.at(i).x);

      const Eigen::Matrix3d MX = dRdx(posevec(0), posevec(2), posevec(1));
      const float arx = (MX * point_ori).dot(coeff_vec);

      const Eigen::Matrix3d MY = dRdy(posevec(0), posevec(2), posevec(1));
      const float ary = (MY * point_ori).dot(coeff_vec);

      const Eigen::Matrix3d MZ = dRdz(posevec(0), posevec(2), posevec(1));
      const float arz = (MZ * point_ori).dot(coeff_vec);

      // lidar -> camera
      A(i, 0) = arz;
      A(i, 1) = arx;
      A(i, 2) = ary;
      A(i, 3) = coeffSel.at(i).x;
      A(i, 4) = coeffSel.at(i).y;
      A(i, 5) = coeffSel.at(i).z;
      b(i) = -intensity;
    }

    const Eigen::MatrixXd AtA = A.transpose() * A;
    const Eigen::VectorXd AtB = A.transpose() * b;

    const Eigen::VectorXd matX = AtA.householderQr().solve(AtB);

    if (iterCount == 0) {
      const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(AtA);
      const Eigen::VectorXd eigenvalues = es.eigenvalues();

      isDegenerate = (eigenvalues.array() < 100.0).any();
    }

    if (!isDegenerate) {
      posevec += matX;
    }

    const float deltaR = rad2deg(matX.head(3)).norm();
    const float deltaT = (100 * matX.tail(3)).norm();

    if (deltaR < 0.05 && deltaT < 0.05) {
      return true; // converged
    }
    return false; // keep optimizing
  }

  void scan2MapOptimization(
    const pcl::PointCloud<PointType> & laserCloudCornerLastDS,
    const pcl::PointCloud<PointType> & laserCloudSurfLastDS,
    const pcl::PointCloud<PointType>::Ptr & laserCloudCornerFromMapDS,
    const pcl::PointCloud<PointType>::Ptr & laserCloudSurfFromMapDS,
    bool & isDegenerate, Vector6d & posevec) const
  {
    if (points3d->empty()) {
      return;
    }

    if (
      edgeFeatureMinValidNum >= static_cast<int>(laserCloudCornerLastDS.size()) ||
      surfFeatureMinValidNum >= static_cast<int>(laserCloudSurfLastDS.size()))
    {
      ROS_WARN(
        "Not enough features! Only %d edge and %d planar features available.",
        laserCloudCornerLastDS.size(), laserCloudSurfLastDS.size());
      return;
    }

    const auto kdtreeCornerFromMap = makeKDTree<PointType>(laserCloudCornerFromMapDS);
    const auto kdtreeSurfFromMap = makeKDTree<PointType>(laserCloudSurfFromMapDS);

    for (int iterCount = 0; iterCount < 30; iterCount++) {
      const auto [laserCloudOri, coeffSel] = optimization(
        laserCloudCornerLastDS, laserCloudSurfLastDS,
        kdtreeCornerFromMap, kdtreeSurfFromMap,
        laserCloudCornerFromMapDS, laserCloudSurfFromMapDS
      );

      if (LMOptimization(laserCloudOri, coeffSel, iterCount, isDegenerate, posevec)) {
        break;
      }
    }

    if (msgIn_->imuAvailable) {
      if (std::abs(msgIn_->initialIMU.y) < 1.4) {
        posevec(0) = interpolate(
          Eigen::Vector3d(posevec(0), 0, 0),
          Eigen::Vector3d(msgIn_->initialIMU.x, 0, 0),
          imuRPYWeight)(0);
        posevec(1) = interpolate(
          Eigen::Vector3d(0, posevec(1), 0),
          Eigen::Vector3d(0, msgIn_->initialIMU.y, 0),
          imuRPYWeight)(1);
      }
    }

    posevec(0) = constraintTransformation(posevec(0), rotation_tolerance);
    posevec(1) = constraintTransformation(posevec(1), rotation_tolerance);
    posevec(5) = constraintTransformation(posevec(5), z_tolerance);
  }

  void saveKeyFramesAndFactor(
    const ros::Time & timestamp,
    const pcl::PointCloud<PointType> & laserCloudCornerLastDS,
    const pcl::PointCloud<PointType> & laserCloudSurfLastDS,
    CornerSurfaceDict & corner_surface_dict)
  {
    if (!points3d->empty()) {
      const Eigen::Affine3d affine0 = getTransformation(makePosevec(poses6dof.back()));
      const Eigen::Affine3d affine1 = getTransformation(posevec);
      const auto [xyz, rpy] = getXYZRPY(affine0.inverse() * affine1);

      if (
        (rpy.array() < surroundingkeyframeAddingAngleThreshold).all() &&
        xyz.norm() < surroundingkeyframeAddingDistThreshold)
      {
        return;
      }
    }

    gtsam::NonlinearFactorGraph gtSAMgraph;

    if (poses6dof.empty()) {
      gtSAMgraph.add(makePriorFactor(posevec));
    } else {
      gtSAMgraph.add(makeOdomFactor(poses6dof, posevec));
    }

    if (
      !points3d->empty() &&
      (poseCovariance(3, 3) >= poseCovThreshold || poseCovariance(4, 4) >= poseCovThreshold))
    {
      const std::optional<gtsam::GPSFactor> gps_factor = gps_factor_.make(
        points3d, gpsCovThreshold, useGpsElevation,
        posevec, last_gps_position, timestamp
      );

      if (gps_factor.has_value()) {
        gtSAMgraph.add(gps_factor.value());
        last_gps_position = gps_factor.value().measurementIn();
        aLoopIsClosed = true;
      }
    }

    // std::cout << "****************************************************" << std::endl;
    // gtSAMgraph.print("GTSAM Graph:\n");

    // update iSAM
    gtsam::Values initial;
    initial.insert(poses6dof.size(), posevecToGtsamPose(posevec));
    isam->update(gtSAMgraph, initial);
    isam->update();

    if (aLoopIsClosed) {
      isam->update();
      isam->update();
      isam->update();
      isam->update();
      isam->update();
    }

    const gtsam::Values estimate = isam->calculateEstimate();
    const gtsam::Pose3 latest = estimate.at<gtsam::Pose3>(estimate.size() - 1);
    // std::cout << "****************************************************" << std::endl;
    // estimate.print("Current estimate: ");

    // size can be used as index
    const PointType position = makePoint(latest.translation(), points3d->size());
    points3d->push_back(position);

    // intensity can be used as index
    const StampedPose pose6dof = makeStampedPose(latest, timestamp.toSec());
    poses6dof.push_back(pose6dof);

    // std::cout << "****************************************************" << std::endl;
    // std::cout << "Pose covariance:" << std::endl;
    // std::cout << isam->marginalCovariance(estimate.size()-1) << std::endl << std::endl;
    poseCovariance = isam->marginalCovariance(estimate.size() - 1);

    // save updated transform
    posevec = getPoseVec(latest);

    // save key frame cloud
    corner_cloud.push_back(laserCloudCornerLastDS);
    surface_cloud.push_back(laserCloudSurfLastDS);

    // save path for visualization
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header.stamp = ros::Time().fromSec(pose6dof.time);
    pose_stamped.header.frame_id = odometryFrame;
    pose_stamped.pose = makePose(latest.rotation().rpy(), latest.translation());
    path_poses_.push_back(pose_stamped);

    if (points3d->empty()) {
      return;
    }

    // correct poses
    if (aLoopIsClosed) {
      // clear map cache
      corner_surface_dict.clear();
      // clear path
      path_poses_.clear();
      // update key poses
      for (unsigned int i = 0; i < estimate.size(); ++i) {
        const gtsam::Pose3 pose = estimate.at<gtsam::Pose3>(i);
        const Eigen::Vector3d xyz = pose.translation();
        const Eigen::Vector3d rpy = pose.rotation().rpy();

        points3d->at(i) = makePoint(xyz, points3d->at(i).intensity);

        const auto point6d = poses6dof.at(i);
        poses6dof.at(i) = makeStampedPose(pose, point6d.time);

        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header.stamp = ros::Time().fromSec(point6d.time);
        pose_stamped.header.frame_id = odometryFrame;
        pose_stamped.pose = makePose(rpy, xyz);

        path_poses_.push_back(pose_stamped);
      }

      aLoopIsClosed = false;
    }
  }

  void publishOdometry(
    const ros::Time & timestamp, const Vector6d & front_posevec,
    const bool isDegenerate, const Vector6d & posevec,
    bool & lastIncreOdomPubFlag, Eigen::Affine3d & increOdomAffine) const
  {
    // Publish odometry for ROS (global)
    nav_msgs::Odometry odometry;
    odometry.header.stamp = timestamp;
    odometry.header.frame_id = odometryFrame;
    odometry.child_frame_id = "odom_mapping";
    odometry.pose.pose = makePose(posevec);
    // geometry_msgs/Quaternion
    pubLaserOdometryGlobal.publish(odometry);

    // Publish TF
    tf::TransformBroadcaster br;
    br.sendTransform(
      tf::StampedTransform(makeTransform(posevec), timestamp, odometryFrame, "lidar_link"));

    nav_msgs::Odometry laserOdomIncremental;
    // Publish odometry for ROS (incremental)
    if (!lastIncreOdomPubFlag) {
      lastIncreOdomPubFlag = true;
      laserOdomIncremental = odometry;
      increOdomAffine = getTransformation(posevec);
    } else {
      const Eigen::Affine3d front = getTransformation(front_posevec);
      Eigen::Affine3d affineIncre = front.inverse() * incrementalOdometryAffineBack;
      increOdomAffine = increOdomAffine * affineIncre;
      Vector6d odometry = getPoseVec(increOdomAffine);
      if (msgIn_->imuAvailable) {
        if (std::abs(msgIn_->initialIMU.y) < 1.4) {
          double imuWeight = 0.1;

          const Eigen::Vector3d r = interpolate(
            Eigen::Vector3d(odometry(0), 0, 0),
            Eigen::Vector3d(msgIn_->initialIMU.x, 0, 0),
            imuWeight
          );
          odometry(0) = r(0);

          const Eigen::Vector3d p = interpolate(
            Eigen::Vector3d(0, odometry(1), 0),
            Eigen::Vector3d(0, msgIn_->initialIMU.y, 0),
            imuWeight
          );
          odometry(1) = p(1);
        }
      }
      laserOdomIncremental.header.stamp = timestamp;
      laserOdomIncremental.header.frame_id = odometryFrame;
      laserOdomIncremental.child_frame_id = "odom_mapping";
      laserOdomIncremental.pose.pose = makePose(odometry);
      if (isDegenerate) {
        laserOdomIncremental.pose.covariance[0] = 1;
      } else {
        laserOdomIncremental.pose.covariance[0] = 0;
      }
    }
    pubLaserOdometryIncremental.publish(laserOdomIncremental);
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
