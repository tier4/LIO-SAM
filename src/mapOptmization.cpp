#include "utility.h"
#include "jacobian.h"
#include "homogeneous.h"
#include "lio_sam/cloud_info.h"
#include "lio_sam/save_map.h"

#include <opencv2/core/eigen.hpp>

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

StampedPose makeStampedPose(
  const Eigen::Vector3d & xyz, const Eigen::Vector3d & rpy,
  const double time)
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

tf::Quaternion tfQuaternionFromRPY(const double roll, const double pitch, const double yaw)
{
  tf::Quaternion q;
  q.setRPY(roll, pitch, yaw);
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
    const auto & point = input.points[i];
    const Eigen::Vector3d p = getXYZ(point);
    output.points[i] = makePoint(transform * p, point.intensity);
  }
  return output;
}

PointType pointAssociateToMap(
  const Eigen::Affine3d & transPointAssociateToMap,
  const PointType & pi)
{
  const Eigen::Vector3d p(pi.x, pi.y, pi.z);
  const Eigen::Vector3d q = transPointAssociateToMap * p;
  return makePoint(q, pi.intensity);
}

bool validatePlane(
  const Points<PointType>::type & points,
  const std::vector<int> & indices,
  const Eigen::Vector4d & x)
{
  for (int j = 0; j < 5; j++) {
    const Eigen::Vector3d p = getXYZ(points.at(indices[j]));
    const Eigen::Vector4d q = toHomogeneous(p);

    if (fabs(x.transpose() * q) > 0.2) {
      return false;
    }
  }
  return true;
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
  const pcl::PointCloud<StampedPose> & cloudKeyPoses6D, const Vector6d & posevec)
{
  const gtsam::Pose3 src = posevecToGtsamPose(makePosevec(cloudKeyPoses6D.points.back()));
  const gtsam::Pose3 dst = posevecToGtsamPose(posevec);

  const Vector6d v = (Vector6d() << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished();
  const auto noise = gtsam::noiseModel::Diagonal::Variances(v);

  const unsigned int size = cloudKeyPoses6D.size();

  return gtsam::BetweenFactor<gtsam::Pose3>(size - 1, size, src.between(dst), noise);
}

std::optional<gtsam::GPSFactor> makeGPSFactor(
  const pcl::PointCloud<PointType> & cloudKeyPoses3D,
  const float gpsCovThreshold, const bool useGpsElevation,
  const Vector6d & posevec, const Eigen::Vector3d & last_gps_position,
  const ros::Time & timestamp,
  std::deque<nav_msgs::Odometry> & gpsQueue)
{

  if (gpsQueue.empty()) {
    return std::nullopt;
  }

  const double distance =
    (getXYZ(cloudKeyPoses3D.front()) - getXYZ(cloudKeyPoses3D.back())).norm();
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
    return std::make_optional<gtsam::GPSFactor>(cloudKeyPoses3D.size(), gps_position, gps_noise);
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
    const pcl::PointCloud<PointType> & cloudKeyPoses3D,
    const float gpsCovThreshold, const bool useGpsElevation,
    const Vector6d & posevec, const Eigen::Vector3d & last_gps_position,
    const ros::Time & timestamp)
  {
    return makeGPSFactor(
      cloudKeyPoses3D, gpsCovThreshold, useGpsElevation,
      posevec, last_gps_position, timestamp, gpsQueue
    );
  }

private:
  std::deque<nav_msgs::Odometry> gpsQueue;
};

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

class mapOptimization : public ParamServer
{
  using CornerSurfaceDict = std::map<
    int,
    std::pair<pcl::PointCloud<PointType>,
    pcl::PointCloud<PointType>>
  >;

public:
  // gtsam
  gtsam::NonlinearFactorGraph gtSAMgraph;
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
  lio_sam::cloud_info cloudInfo;

  std::vector<pcl::PointCloud<PointType>> corner_cloud;
  std::vector<pcl::PointCloud<PointType>> surface_cloud;

  pcl::PointCloud<PointType> cloudKeyPoses3D;
  pcl::PointCloud<StampedPose> cloudKeyPoses6D;

  CornerSurfaceDict corner_surface_dict;
  pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMapDS;
  pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS;

  pcl::KdTreeFLANN<PointType> kdtreeCornerFromMap;
  pcl::KdTreeFLANN<PointType> kdtreeSurfFromMap;

  ros::Time timestamp;

  std::mutex mtx;

  bool isDegenerate;
  bool aLoopIsClosed;

  std::vector<geometry_msgs::PoseStamped> path_poses_;

  Eigen::Affine3d incrementalOdometryAffineFront;
  Eigen::Affine3d incrementalOdometryAffineBack;
  Eigen::Affine3d lastImuTransformation;
  Eigen::Vector3d last_gps_position;

  bool lastImuPreTransAvailable;
  Eigen::Affine3d lastImuPreTransformation;

  bool lastIncreOdomPubFlag;
  nav_msgs::Odometry laserOdomIncremental; // incremental odometry msg
  Eigen::Affine3d increOdomAffine; // incremental odometry in affine

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
    isDegenerate(false),
    aLoopIsClosed(false),
    lastImuPreTransAvailable(false),
    lastIncreOdomPubFlag(false)
  {

    laserCloudCornerFromMapDS.reset(new pcl::PointCloud<PointType>());
    laserCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>());
  }

  void laserCloudInfoHandler(const lio_sam::cloud_infoConstPtr & msgIn)
  {
    // extract time stamp
    timestamp = msgIn->header.stamp;

    // extract info and feature cloud
    cloudInfo = *msgIn;

    // corner feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudCornerLast(new pcl::PointCloud<PointType>());

    // surf feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast(new pcl::PointCloud<PointType>());

    pcl::fromROSMsg(msgIn->cloud_corner, *laserCloudCornerLast);
    pcl::fromROSMsg(msgIn->cloud_surface, *laserCloudSurfLast);

    std::lock_guard<std::mutex> lock(mtx);

    static double timeLastProcessing = -1;
    if (timestamp.toSec() - timeLastProcessing >= mappingProcessInterval) {
      timeLastProcessing = timestamp.toSec();

      // save current transformation before any processing
      Eigen::Affine3d incrementalOdometryAffineFront = getTransformation(posevec);
      updateInitialGuess();

      extractSurroundingKeyFrames(cloudKeyPoses6D, corner_surface_dict);

      pcl::VoxelGrid<PointType> downSizeFilterCorner;
      downSizeFilterCorner.setLeafSize(
        mappingCornerLeafSize,
        mappingCornerLeafSize,
        mappingCornerLeafSize);
      pcl::PointCloud<PointType> laserCloudCornerLastDS;
      downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
      downSizeFilterCorner.filter(laserCloudCornerLastDS);

      pcl::VoxelGrid<PointType> downSizeFilterSurf;
      downSizeFilterSurf.setLeafSize(
        mappingSurfLeafSize,
        mappingSurfLeafSize,
        mappingSurfLeafSize);
      pcl::PointCloud<PointType> laserCloudSurfLastDS;
      downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
      downSizeFilterSurf.filter(laserCloudSurfLastDS);

      scan2MapOptimization(laserCloudCornerLastDS, laserCloudSurfLastDS);

      gtsam::Values isamCurrentEstimate;
      saveKeyFramesAndFactor(laserCloudCornerLastDS, laserCloudSurfLastDS, isamCurrentEstimate);

      correctPoses(isamCurrentEstimate, corner_surface_dict);

      publishOdometry(incrementalOdometryAffineFront);

      publishFrames(laserCloudCornerLastDS, laserCloudSurfLastDS);
    }
  }

  void updateInitialGuess()
  {

    const Eigen::Vector3d rpy = vector3ToEigen(cloudInfo.initialIMU);

    // initialization
    if (cloudKeyPoses3D.points.empty()) {
      posevec.head(3) = vector3ToEigen(cloudInfo.initialIMU);

      if (!useImuHeadingInitialization) {
        posevec(2) = 0;
      }

      lastImuTransformation = makeAffine(rpy, Eigen::Vector3d::Zero());
      // save imu before return;
      return;
    }

    // use imu pre-integration estimation for pose guess
    if (cloudInfo.odomAvailable) {
      const Eigen::Affine3d back = poseToAffine(cloudInfo.initial_pose);
      if (!lastImuPreTransAvailable) {
        lastImuPreTransformation = back;
        lastImuPreTransAvailable = true;
      } else {
        const Eigen::Affine3d incre = lastImuPreTransformation.inverse() * back;
        const Eigen::Affine3d tobe = getTransformation(posevec);
        posevec = getPoseVec(tobe * incre);

        lastImuPreTransformation = back;

        // save imu before return;
        lastImuTransformation = makeAffine(rpy, Eigen::Vector3d::Zero());
        return;
      }
    }

    // use imu incremental estimation for pose guess (only rotation)
    if (cloudInfo.imuAvailable) {
      const Eigen::Affine3d back = makeAffine(rpy, Eigen::Vector3d::Zero());
      const Eigen::Affine3d incre = lastImuTransformation.inverse() * back;

      const Eigen::Affine3d tobe = getTransformation(posevec);
      posevec = getPoseVec(tobe * incre);

      // save imu before return;
      lastImuTransformation = makeAffine(rpy, Eigen::Vector3d::Zero());
      return;
    }
  }

  void extractSurroundingKeyFrames(
    const pcl::PointCloud<StampedPose> & cloudKeyPoses6D,
    CornerSurfaceDict & corner_surface_dict)
  {
    if (cloudKeyPoses3D.points.empty()) {
      return;
    }

    pcl::PointCloud<PointType>::Ptr poses(new pcl::PointCloud<PointType>());
    std::vector<int> indices;
    std::vector<float> pointSearchSqDis;

    pcl::KdTreeFLANN<PointType> kdtree;

    // extract all the nearby key poses and downsample them
    kdtree.setInputCloud(cloudKeyPoses3D.makeShared()); // create kd-tree
    kdtree.radiusSearch(
      cloudKeyPoses3D.back(),
      (double)surroundingKeyframeSearchRadius, indices, pointSearchSqDis);
    for (unsigned int index : indices) {
      poses->push_back(cloudKeyPoses3D.points[index]);
    }

    pcl::PointCloud<PointType> downsampled = downsample(poses, surroundingKeyframeDensity);
    for (auto & pt : downsampled.points) {
      kdtree.nearestKSearch(pt, 1, indices, pointSearchSqDis);
      pt.intensity = cloudKeyPoses3D.points[indices[0]].intensity;
    }

    // also extract some latest key frames in case the robot rotates in one position
    for (int i = cloudKeyPoses3D.size() - 1; i >= 0; --i) {
      if (timestamp.toSec() - cloudKeyPoses6D.points[i].time < 10.0) {
        downsampled.push_back(cloudKeyPoses3D.points[i]);
      } else {
        break;
      }
    }

    // fuse the map
    pcl::PointCloud<PointType>::Ptr corner(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr surface(new pcl::PointCloud<PointType>());

    for (unsigned int i = 0; i < downsampled.size(); ++i) {
      const double distance =
        (getXYZ(downsampled.points[i]) - getXYZ(cloudKeyPoses3D.back())).norm();
      if (distance > surroundingKeyframeSearchRadius) {
        continue;
      }

      int index = static_cast<int>(downsampled.points[i].intensity);
      if (corner_surface_dict.find(index) != corner_surface_dict.end()) {
        // transformed cloud available
        *corner += corner_surface_dict[index].first;
        *surface += corner_surface_dict[index].second;
        continue;
      }
      // transformed cloud not available
      pcl::PointCloud<PointType> c = transform(
        corner_cloud[index], makePosevec(cloudKeyPoses6D.points[index]));
      pcl::PointCloud<PointType> s = transform(
        surface_cloud[index], makePosevec(cloudKeyPoses6D.points[index]));
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
    const pcl::PointCloud<PointType> & laserCloudSurfLastDS) const
  {
    const Eigen::Affine3d transPointAssociateToMap = getTransformation(posevec);
    std::vector<PointType> laserCloudOriCornerVec(N_SCAN * Horizon_SCAN);
    std::vector<PointType> coeffSelCornerVec(N_SCAN * Horizon_SCAN);
    // corner point holder for parallel computation
    std::vector<bool> laserCloudOriCornerFlag(N_SCAN * Horizon_SCAN, false);

    // corner optimization
    #pragma omp parallel for num_threads(numberOfCores)
    for (unsigned int i = 0; i < laserCloudCornerLastDS.size(); i++) {
      std::vector<int> indices;
      std::vector<float> pointSearchSqDis;

      const PointType pointOri = laserCloudCornerLastDS.points[i];
      const PointType pointSel = pointAssociateToMap(transPointAssociateToMap, pointOri);
      kdtreeCornerFromMap.nearestKSearch(pointSel, 5, indices, pointSearchSqDis);

      if (pointSearchSqDis[4] < 1.0) {
        Eigen::Vector3d c = Eigen::Vector3d::Zero();
        for (int j = 0; j < 5; j++) {
          c += getXYZ(laserCloudCornerFromMapDS->points[indices[j]]);
        }
        c /= 5.0;

        Eigen::Matrix3d sa = Eigen::Matrix3d::Zero();

        for (int j = 0; j < 5; j++) {
          const Eigen::Vector3d x = getXYZ(laserCloudCornerFromMapDS->points[indices[j]]);
          const Eigen::Vector3d a = x - c;
          sa += a * a.transpose();
        }

        sa = sa / 5.0;

        cv::Mat matA1(3, 3, CV_64F, cv::Scalar::all(0));
        cv::eigen2cv(sa, matA1);
        cv::Mat matD1(1, 3, CV_64F, cv::Scalar::all(0));
        cv::Mat matV1(3, 3, CV_64F, cv::Scalar::all(0));
        cv::eigen(matA1, matD1, matV1);
        Eigen::Matrix3d v1;
        cv::cv2eigen(matV1, v1);

        if (matD1.at<float>(0, 0) > 3 * matD1.at<float>(0, 1)) {
          const Eigen::Vector3d p0 = getXYZ(pointSel);
          const Eigen::Vector3d p1 = c + 0.1 * v1.row(0).transpose();
          const Eigen::Vector3d p2 = c - 0.1 * v1.row(0).transpose();

          const Eigen::Vector3d d01 = p0 - p1;
          const Eigen::Vector3d d02 = p0 - p2;
          const Eigen::Vector3d d12 = p1 - p2;

          // const Eigen::Vector3d d012(d01(0) * d02(1) - d02(0) * d01(1),
          //                            d01(0) * d02(2) - d02(0) * d01(2),
          //                            d01(1) * d02(2) - d02(1) * d01(2));
          const Eigen::Vector3d cross(d01(1) * d02(2) - d01(2) * d02(1),
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
            laserCloudOriCornerVec[i] = pointOri;
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

      const PointType pointOri = laserCloudSurfLastDS.points[i];
      const PointType pointSel = pointAssociateToMap(transPointAssociateToMap, pointOri);
      kdtreeSurfFromMap.nearestKSearch(pointSel, 5, indices, squared_distances);

      Eigen::Matrix<double, 5, 3> matA0;
      Eigen::Matrix<double, 5, 1> matB0;

      matA0.setZero();
      matB0.fill(-1);

      if (squared_distances[4] >= 1.0) {
        continue;
      }

      for (int j = 0; j < 5; j++) {
        matA0.row(j) = getXYZ(laserCloudSurfFromMapDS->points[indices[j]]);
      }

      const Eigen::Vector3d matX0 = matA0.colPivHouseholderQr().solve(matB0);

      const Eigen::Vector4d x = toHomogeneous(matX0) / matX0.norm();

      if (!validatePlane(laserCloudSurfFromMapDS->points, indices, x)) {
        continue;
      }

      const Eigen::Vector3d p = getXYZ(pointSel);
      const Eigen::Vector4d q = toHomogeneous(p);
      const float pd2 = x.transpose() * q;
      const float s = 1 - 0.9 * fabs(pd2) / sqrt(p.norm());

      if (s <= 0.1) {
        continue;
      }

      laserCloudOriSurfVec[i] = pointOri;
      coeffSelSurfVec[i] = makePoint(s * x.head(3), s * pd2);
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
    const int iterCount)
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
      const float intensity = coeffSel.points[i].intensity;

      // in camera

      const Eigen::Vector3d point_ori(
        laserCloudOri.points[i].y,
        laserCloudOri.points[i].z,
        laserCloudOri.points[i].x);

      const Eigen::Vector3d coeff_vec(
        coeffSel.points[i].y,
        coeffSel.points[i].z,
        coeffSel.points[i].x);

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
      A(i, 3) = coeffSel.points[i].x;
      A(i, 4) = coeffSel.points[i].y;
      A(i, 5) = coeffSel.points[i].z;
      b(i) = -intensity;
    }

    const Eigen::MatrixXd AtA = A.transpose() * A;
    const Eigen::VectorXd AtB = A.transpose() * b;

    const Eigen::VectorXd matX = AtA.householderQr().solve(AtB);

    if (iterCount == 0) {
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(AtA);
      const Eigen::VectorXd eigenvalues = es.eigenvalues();

      isDegenerate = (eigenvalues.array() < 100.0).any();
    }

    if (!isDegenerate) {
      posevec += matX;
    }

    float deltaR = sqrt(
      pow(pcl::rad2deg(matX(0)), 2) +
      pow(pcl::rad2deg(matX(1)), 2) +
      pow(pcl::rad2deg(matX(2)), 2));
    float deltaT = sqrt(
      pow(matX(3) * 100, 2) +
      pow(matX(4) * 100, 2) +
      pow(matX(5) * 100, 2));

    if (deltaR < 0.05 && deltaT < 0.05) {
      return true; // converged
    }
    return false; // keep optimizing
  }

  void scan2MapOptimization(
    const pcl::PointCloud<PointType> & laserCloudCornerLastDS,
    const pcl::PointCloud<PointType> & laserCloudSurfLastDS)
  {
    if (cloudKeyPoses3D.points.empty()) {
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

    kdtreeCornerFromMap.setInputCloud(laserCloudCornerFromMapDS);
    kdtreeSurfFromMap.setInputCloud(laserCloudSurfFromMapDS);

    for (int iterCount = 0; iterCount < 30; iterCount++) {
      const auto [laserCloudOri, coeffSel] = optimization(
        laserCloudCornerLastDS, laserCloudSurfLastDS
      );

      if (LMOptimization(laserCloudOri, coeffSel, iterCount)) {
        break;
      }
    }

    if (cloudInfo.imuAvailable) {
      if (std::abs(cloudInfo.initialIMU.y) < 1.4) {
        // slerp roll
        const tf::Quaternion qr0 = tfQuaternionFromRPY(posevec(0), 0, 0);
        const tf::Quaternion qr1 = tfQuaternionFromRPY(cloudInfo.initialIMU.x, 0, 0);
        posevec(0) = getRPY(interpolate(qr0, qr1, imuRPYWeight))(0);

        // slerp pitch
        const tf::Quaternion qp0 = tfQuaternionFromRPY(0, posevec(1), 0);
        const tf::Quaternion qp1 = tfQuaternionFromRPY(0, cloudInfo.initialIMU.y, 0);
        posevec(1) = getRPY(interpolate(qp0, qp1, imuRPYWeight))(1);
      }
    }

    posevec(0) = constraintTransformation(posevec(0), rotation_tolerance);
    posevec(1) = constraintTransformation(posevec(1), rotation_tolerance);
    posevec(5) = constraintTransformation(posevec(5), z_tolerance);

    incrementalOdometryAffineBack = getTransformation(posevec);
  }

  void addGPSFactor()
  {
    const std::optional<gtsam::GPSFactor> gps_factor = gps_factor_.make(
      cloudKeyPoses3D, gpsCovThreshold, useGpsElevation,
      posevec, last_gps_position, timestamp
    );

    if (!gps_factor.has_value()) {
      return;
    }

    gtSAMgraph.add(gps_factor.value());
    last_gps_position = gps_factor.value().measurementIn();
    aLoopIsClosed = true;
  }

  void saveKeyFramesAndFactor(
    const pcl::PointCloud<PointType> & laserCloudCornerLastDS,
    const pcl::PointCloud<PointType> & laserCloudSurfLastDS,
    gtsam::Values & isamCurrentEstimate)
  {
    if (!cloudKeyPoses3D.points.empty()) {
      Eigen::Affine3d transStart = getTransformation(makePosevec(cloudKeyPoses6D.back()));
      Eigen::Affine3d transFinal = getTransformation(posevec);
      const auto [xyz, rpy] = getXYZRPY(transStart.inverse() * transFinal);

      if (
        abs(rpy(0)) < surroundingkeyframeAddingAngleThreshold &&
        abs(rpy(1)) < surroundingkeyframeAddingAngleThreshold &&
        abs(rpy(2)) < surroundingkeyframeAddingAngleThreshold &&
        xyz.norm() < surroundingkeyframeAddingDistThreshold)
      {
        return;
      }
    }

    if (cloudKeyPoses6D.empty()) {
      gtSAMgraph.add(makePriorFactor(posevec));
    } else {
      gtSAMgraph.add(makeOdomFactor(cloudKeyPoses6D, posevec));
    }

    if (
      !cloudKeyPoses3D.points.empty() &&
      (poseCovariance(3, 3) >= poseCovThreshold || poseCovariance(4, 4) >= poseCovThreshold))
    {
      addGPSFactor();
    }

    // std::cout << "****************************************************" << std::endl;
    // gtSAMgraph.print("GTSAM Graph:\n");

    // update iSAM
    gtsam::Values initial;
    initial.insert(cloudKeyPoses6D.size(), posevecToGtsamPose(posevec));
    isam->update(gtSAMgraph, initial);
    isam->update();

    if (aLoopIsClosed) {
      isam->update();
      isam->update();
      isam->update();
      isam->update();
      isam->update();
    }

    gtSAMgraph.resize(0);

    //save key poses
    gtsam::Pose3 latestEstimate;

    isamCurrentEstimate = isam->calculateEstimate();
    latestEstimate = isamCurrentEstimate.at<gtsam::Pose3>(isamCurrentEstimate.size() - 1);
    // std::cout << "****************************************************" << std::endl;
    // isamCurrentEstimate.print("Current estimate: ");

    // size can be used as index
    const PointType position = makePoint(latestEstimate.translation(), cloudKeyPoses3D.size());
    cloudKeyPoses3D.push_back(position);

    const Eigen::Vector3d xyz = latestEstimate.translation();
    const Eigen::Vector3d rpy = latestEstimate.rotation().rpy();
    // intensity can be used as index
    const StampedPose pose6dof = makeStampedPose(xyz, rpy, timestamp.toSec());
    cloudKeyPoses6D.push_back(pose6dof);

    // std::cout << "****************************************************" << std::endl;
    // std::cout << "Pose covariance:" << std::endl;
    // std::cout << isam->marginalCovariance(isamCurrentEstimate.size()-1) << std::endl << std::endl;
    poseCovariance = isam->marginalCovariance(isamCurrentEstimate.size() - 1);

    // save updated transform
    posevec = getPoseVec(latestEstimate);

    // save key frame cloud
    corner_cloud.push_back(laserCloudCornerLastDS);
    surface_cloud.push_back(laserCloudSurfLastDS);

    // save path for visualization
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header.stamp = ros::Time().fromSec(pose6dof.time);
    pose_stamped.header.frame_id = odometryFrame;
    pose_stamped.pose = makePose(xyz, rpy);
    path_poses_.push_back(pose_stamped);
  }

  void correctPoses(
    const gtsam::Values & isamCurrentEstimate, CornerSurfaceDict & corner_surface_dict)
  {
    if (cloudKeyPoses3D.points.empty()) {
      return;
    }

    if (aLoopIsClosed) {
      // clear map cache
      corner_surface_dict.clear();
      // clear path
      path_poses_.clear();
      // update key poses
      for (unsigned int i = 0; i < isamCurrentEstimate.size(); ++i) {
        const Eigen::Vector3d xyz = isamCurrentEstimate.at<gtsam::Pose3>(i).translation();
        const Eigen::Vector3d rpy = isamCurrentEstimate.at<gtsam::Pose3>(i).rotation().rpy();

        const auto point3d = cloudKeyPoses3D.points[i];
        cloudKeyPoses3D.points[i] = makePoint(xyz, point3d.intensity);

        const auto point6d = cloudKeyPoses6D.points[i];
        cloudKeyPoses6D.points[i] = makeStampedPose(xyz, rpy, point6d.time);

        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header.stamp = ros::Time().fromSec(point6d.time);
        pose_stamped.header.frame_id = odometryFrame;
        pose_stamped.pose = makePose(xyz, rpy);

        path_poses_.push_back(pose_stamped);
      }

      aLoopIsClosed = false;
    }
  }

  void publishOdometry(const Eigen::Affine3d & incrementalOdometryAffineFront)
  {
    // Publish odometry for ROS (global)
    nav_msgs::Odometry laserOdometryROS;
    laserOdometryROS.header.stamp = timestamp;
    laserOdometryROS.header.frame_id = odometryFrame;
    laserOdometryROS.child_frame_id = "odom_mapping";
    laserOdometryROS.pose.pose = makePose(posevec);
    // geometry_msgs/Quaternion
    pubLaserOdometryGlobal.publish(laserOdometryROS);

    // Publish TF
    tf::TransformBroadcaster br;
    tf::Transform t_odom_to_lidar = makeTransform(posevec);
    tf::StampedTransform trans_odom_to_lidar = tf::StampedTransform(
      t_odom_to_lidar, timestamp, odometryFrame, "lidar_link");
    br.sendTransform(trans_odom_to_lidar);

    // Publish odometry for ROS (incremental)
    if (!lastIncreOdomPubFlag) {
      lastIncreOdomPubFlag = true;
      laserOdomIncremental = laserOdometryROS;
      increOdomAffine = getTransformation(posevec);
    } else {
      Eigen::Affine3d affineIncre = incrementalOdometryAffineFront.inverse() *
        incrementalOdometryAffineBack;
      increOdomAffine = increOdomAffine * affineIncre;
      Vector6d odometry = getPoseVec(increOdomAffine);
      if (cloudInfo.imuAvailable) {
        if (std::abs(cloudInfo.initialIMU.y) < 1.4) {
          double imuWeight = 0.1;
          tf::Quaternion imuQuaternion;
          tf::Quaternion transformQuaternion;

          // slerp roll
          transformQuaternion.setRPY(odometry(0), 0, 0);
          imuQuaternion.setRPY(cloudInfo.initialIMU.x, 0, 0);
          odometry(0) = getRPY(interpolate(transformQuaternion, imuQuaternion, imuWeight))(0);

          // slerp pitch
          transformQuaternion.setRPY(0, odometry(1), 0);
          imuQuaternion.setRPY(0, cloudInfo.initialIMU.y, 0);
          odometry(1) = getRPY(interpolate(transformQuaternion, imuQuaternion, imuWeight))(1);
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

  void publishFrames(
    const pcl::PointCloud<PointType> & laserCloudCornerLastDS,
    const pcl::PointCloud<PointType> & laserCloudSurfLastDS) const
  {
    if (cloudKeyPoses3D.points.empty()) {
      return;
    }
    // publish key poses
    publishCloud(pubKeyPoses, cloudKeyPoses3D, timestamp, odometryFrame);
    // Publish surrounding key frames
    publishCloud(
      pubRecentKeyFrames, *laserCloudSurfFromMapDS, timestamp,
      odometryFrame);
    // publish registered key frame
    if (pubRecentKeyFrame.getNumSubscribers() != 0) {
      pcl::PointCloud<PointType> cloudOut;
      cloudOut += transform(laserCloudCornerLastDS, posevec);
      cloudOut += transform(laserCloudSurfLastDS, posevec);
      publishCloud(pubRecentKeyFrame, cloudOut, timestamp, odometryFrame);
    }
    publishPath(pubPath, odometryFrame, timestamp, path_poses_);
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
