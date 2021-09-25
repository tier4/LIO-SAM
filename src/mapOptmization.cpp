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

using namespace gtsam;

using symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)
using symbol_shorthand::G; // GPS pose

typedef Eigen::Matrix<double, 6, 1> Vector6d;

/*
    * A point cloud type that has 6D pose info ([x,y,z,roll,pitch,yaw] intensity is time stamp)
    */
struct PointXYZIRPYT
{
  PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;                // preferred way of adding a XYZ+padding
  float roll;
  float pitch;
  float yaw;
  double time;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT(
  PointXYZIRPYT,
  (float, x, x)(float, y, y)(float, z, z) (float, intensity, intensity)(float, roll, roll) (
    float,
    pitch, pitch) (float, yaw, yaw)(double, time, time))

typedef PointXYZIRPYT PointTypePose;

Eigen::Vector3f getRPY(const tf::Quaternion & q)
{
  double roll, pitch, yaw;
  tf::Matrix3x3(q).getRPY(roll, pitch, yaw);
  return Eigen::Vector3f((float)roll, (float)pitch, (float)yaw);
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

geometry_msgs::Pose makePose(const Vector6d & posevec)
{
  geometry_msgs::Pose pose;
  pose.position.x = posevec(3);
  pose.position.y = posevec(4);
  pose.position.z = posevec(5);
  pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(
    posevec(0), posevec(1), posevec(2)
  );
  return pose;
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

Eigen::Affine3d getTransformation(const Vector6d & posevec)
{
  Eigen::Affine3d transform;
  pcl::getTransformation(
    posevec(3), posevec(4), posevec(5),
    posevec(0), posevec(1), posevec(2));
  return transform;
}

gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint)
{
  return gtsam::Pose3(
    gtsam::Rot3::RzRyRx(
      double(thisPoint.roll),
      double(thisPoint.pitch),
      double(thisPoint.yaw)),
    gtsam::Point3(
      double(thisPoint.x),
      double(thisPoint.y),
      double(thisPoint.z)));
}

gtsam::Pose3 trans2gtsamPose(const Vector6d & transformIn)
{
  return gtsam::Pose3(
    gtsam::Rot3::RzRyRx(transformIn(0), transformIn(1), transformIn(2)),
    gtsam::Point3(transformIn(3), transformIn(4), transformIn(5)));
}

Eigen::Affine3d pclPointToAffine3d(PointTypePose thisPoint)
{
  Eigen::Affine3d transform;
  pcl::getTransformation(
    thisPoint.x, thisPoint.y, thisPoint.z,
    thisPoint.roll, thisPoint.pitch, thisPoint.yaw,
    transform);
  return transform;
}

Eigen::Affine3d trans2Affine3d(const Vector6d & transformIn)
{
  Eigen::Affine3d transform;
  pcl::getTransformation(
    transformIn(3), transformIn(4), transformIn(5),
    transformIn(0), transformIn(1), transformIn(2),
    transform);
  return transform;
}

PointTypePose trans2PointTypePose(const Vector6d & transformIn)
{
  PointTypePose thisPose6D;
  thisPose6D.x = transformIn(3);
  thisPose6D.y = transformIn(4);
  thisPose6D.z = transformIn(5);
  thisPose6D.roll = transformIn(0);
  thisPose6D.pitch = transformIn(1);
  thisPose6D.yaw = transformIn(2);
  return thisPose6D;
}

pcl::PointCloud<PointType> transformPointCloud(
  const pcl::PointCloud<PointType> & cloudIn, const PointTypePose & transformIn,
  const int numberOfCores = 2)
{
  pcl::PointCloud<PointType> cloudOut;

  cloudOut.resize(cloudIn.size());

  const Eigen::Affine3d transCur = pclPointToAffine3d(transformIn);

  #pragma omp parallel for num_threads(numberOfCores)
  for (int i = 0; i < cloudIn.size(); ++i) {
    const auto & pointFrom = cloudIn.points[i];
    const Eigen::Vector3d p(pointFrom.x, pointFrom.y, pointFrom.z);
    cloudOut.points[i] = makePoint(transCur * p, pointFrom.intensity);
  }
  return cloudOut;
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

class mapOptimization : public ParamServer
{

public:
  // gtsam
  NonlinearFactorGraph gtSAMgraph;
  Values initialEstimate;
  Values isamCurrentEstimate;
  Eigen::MatrixXd poseCovariance;

  const ros::Publisher pubLaserCloudSurround;
  const ros::Publisher pubLaserOdometryGlobal;
  const ros::Publisher pubLaserOdometryIncremental;
  const ros::Publisher pubKeyPoses;
  const ros::Publisher pubPath;

  const ros::Publisher pubRecentKeyFrames;
  const ros::Publisher pubRecentKeyFrame;
  const ros::Publisher pubCloudRegisteredRaw;

  const ros::Subscriber subCloud;
  const ros::Subscriber subGPS;

  std::shared_ptr<ISAM2> isam;

  std::deque<nav_msgs::Odometry> gpsQueue;
  lio_sam::cloud_info cloudInfo;

  std::vector<pcl::PointCloud<PointType>> cornerCloudKeyFrames;
  std::vector<pcl::PointCloud<PointType>> surfCloudKeyFrames;

  pcl::PointCloud<PointType> cloudKeyPoses3D;
  pcl::PointCloud<PointTypePose> cloudKeyPoses6D;
  pcl::PointCloud<PointType>::Ptr copy_cloudKeyPoses3D;
  pcl::PointCloud<PointTypePose>::Ptr copy_cloudKeyPoses6D;

  // corner feature set from odoOptimization
  pcl::PointCloud<PointType>::Ptr laserCloudCornerLast;
  // surf feature set from odoOptimization
  pcl::PointCloud<PointType>::Ptr laserCloudSurfLast;

  pcl::PointCloud<PointType>::Ptr laserCloudOri;
  pcl::PointCloud<PointType>::Ptr coeffSel;

  // corner point holder for parallel computation
  std::vector<PointType> laserCloudOriCornerVec;
  std::vector<PointType> coeffSelCornerVec;
  std::vector<bool> laserCloudOriCornerFlag;
  // surf point holder for parallel computation
  std::vector<PointType> laserCloudOriSurfVec;
  std::vector<PointType> coeffSelSurfVec;
  std::vector<bool> laserCloudOriSurfFlag;

  std::map<int,
    std::pair<pcl::PointCloud<PointType>, pcl::PointCloud<PointType>>> laserCloudMapContainer;
  pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMapDS;
  pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS;

  pcl::KdTreeFLANN<PointType> kdtreeCornerFromMap;
  pcl::KdTreeFLANN<PointType> kdtreeSurfFromMap;

  pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses;
  pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses;

  pcl::VoxelGrid<PointType> downSizeFilterCorner;
  pcl::VoxelGrid<PointType> downSizeFilterSurf;
  pcl::VoxelGrid<PointType> downSizeFilterICP;
  // for surrounding key poses of scan-to-map optimization
  pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses;

  ros::Time timeLaserInfoStamp;
  double timeLaserInfoCur;

  Vector6d posevec;

  std::mutex mtx;

  bool isDegenerate = false;

  int laserCloudCornerFromMapDSNum = 0;
  int laserCloudSurfFromMapDSNum = 0;

  bool aLoopIsClosed = false;

  nav_msgs::Path globalPath;

  Eigen::Affine3d incrementalOdometryAffineFront;
  Eigen::Affine3d incrementalOdometryAffineBack;

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
    pubCloudRegisteredRaw(
      nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/cloud_registered_raw", 1)),
    subCloud(nh.subscribe<lio_sam::cloud_info>(
        "lio_sam/feature/cloud_info", 1, &mapOptimization::laserCloudInfoHandler,
        this, ros::TransportHints().tcpNoDelay())),
    subGPS(nh.subscribe<nav_msgs::Odometry>(
        gpsTopic, 200, &mapOptimization::gpsHandler, this,
        ros::TransportHints().tcpNoDelay())),
    isam(std::make_shared<ISAM2>(gtsam::ISAM2Params(gtsam::ISAM2GaussNewtonParams(), 0.1, 1)))
  {
    downSizeFilterCorner.setLeafSize(
      mappingCornerLeafSize, mappingCornerLeafSize,
      mappingCornerLeafSize);
    downSizeFilterSurf.setLeafSize(
      mappingSurfLeafSize, mappingSurfLeafSize,
      mappingSurfLeafSize);
    downSizeFilterICP.setLeafSize(
      mappingSurfLeafSize, mappingSurfLeafSize,
      mappingSurfLeafSize);
    downSizeFilterSurroundingKeyPoses.setLeafSize(
      surroundingKeyframeDensity,
      surroundingKeyframeDensity,
      surroundingKeyframeDensity);   // for surrounding key poses of scan-to-map optimization

    allocateMemory();
  }

  void allocateMemory()
  {
    copy_cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
    copy_cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

    kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
    kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());

    // corner feature set from odoOptimization
    laserCloudCornerLast.reset(new pcl::PointCloud<PointType>());

    // surf feature set from odoOptimization
    laserCloudSurfLast.reset(new pcl::PointCloud<PointType>());

    laserCloudOri.reset(new pcl::PointCloud<PointType>());
    coeffSel.reset(new pcl::PointCloud<PointType>());

    laserCloudOriCornerVec.resize(N_SCAN * Horizon_SCAN);
    coeffSelCornerVec.resize(N_SCAN * Horizon_SCAN);
    laserCloudOriCornerFlag.resize(N_SCAN * Horizon_SCAN);
    laserCloudOriSurfVec.resize(N_SCAN * Horizon_SCAN);
    coeffSelSurfVec.resize(N_SCAN * Horizon_SCAN);
    laserCloudOriSurfFlag.resize(N_SCAN * Horizon_SCAN);

    std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
    std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);

    laserCloudCornerFromMapDS.reset(new pcl::PointCloud<PointType>());
    laserCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>());

    for (int i = 0; i < 6; ++i) {
      posevec(i) = 0;
    }
  }

  void laserCloudInfoHandler(const lio_sam::cloud_infoConstPtr & msgIn)
  {
    // extract time stamp
    timeLaserInfoStamp = msgIn->header.stamp;
    timeLaserInfoCur = msgIn->header.stamp.toSec();

    // extract info and feature cloud
    cloudInfo = *msgIn;
    pcl::fromROSMsg(msgIn->cloud_corner, *laserCloudCornerLast);
    pcl::fromROSMsg(msgIn->cloud_surface, *laserCloudSurfLast);

    std::lock_guard<std::mutex> lock(mtx);

    static double timeLastProcessing = -1;
    if (timeLaserInfoCur - timeLastProcessing >= mappingProcessInterval) {
      timeLastProcessing = timeLaserInfoCur;

      updateInitialGuess();

      extractSurroundingKeyFrames();

      pcl::PointCloud<PointType> laserCloudCornerLastDS;
      downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
      downSizeFilterCorner.filter(laserCloudCornerLastDS);

      pcl::PointCloud<PointType> laserCloudSurfLastDS;
      downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
      downSizeFilterSurf.filter(laserCloudSurfLastDS);

      scan2MapOptimization(laserCloudCornerLastDS, laserCloudSurfLastDS);

      saveKeyFramesAndFactor(laserCloudCornerLastDS, laserCloudSurfLastDS);

      correctPoses();

      publishOdometry();

      publishFrames(laserCloudCornerLastDS, laserCloudSurfLastDS);
    }
  }

  void gpsHandler(const nav_msgs::Odometry::ConstPtr & gpsMsg)
  {
    gpsQueue.push_back(*gpsMsg);
  }

  void visualizeGlobalMapThread()
  {
    ros::Rate rate(0.2);
    while (ros::ok()) {
      rate.sleep();
      publishGlobalMap();
    }
  }

  void publishGlobalMap()
  {
    if (pubLaserCloudSurround.getNumSubscribers() == 0) {
      return;
    }

    if (cloudKeyPoses3D.points.empty()) {
      return;
    }

    pcl::KdTreeFLANN<PointType> kdtreeGlobalMap;
    pcl::PointCloud<PointType>::Ptr globalMapKeyPoses(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType> globalMapKeyPosesDS;
    pcl::PointCloud<PointType>::Ptr globalMapKeyFrames(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType> globalMapKeyFramesDS;

    // kd-tree to find near key frames to visualize
    std::vector<int> pointSearchIndGlobalMap;
    std::vector<float> pointSearchSqDisGlobalMap;
    // search near key frames to visualize
    mtx.lock();
    kdtreeGlobalMap.setInputCloud(cloudKeyPoses3D.makeShared());
    kdtreeGlobalMap.radiusSearch(
      cloudKeyPoses3D.back(),
      globalMapVisualizationSearchRadius, pointSearchIndGlobalMap,
      pointSearchSqDisGlobalMap, 0);
    mtx.unlock();

    for (int i = 0; i < (int)pointSearchIndGlobalMap.size(); ++i) {
      globalMapKeyPoses->push_back(
        cloudKeyPoses3D.points[pointSearchIndGlobalMap[i]]);
    }
    // downsample near selected key frames
    pcl::VoxelGrid<PointType>
    downSizeFilterGlobalMapKeyPoses; // for global map visualization
    downSizeFilterGlobalMapKeyPoses.setLeafSize(
      globalMapVisualizationPoseDensity,
      globalMapVisualizationPoseDensity,
      globalMapVisualizationPoseDensity);   // for global map visualization
    downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
    downSizeFilterGlobalMapKeyPoses.filter(globalMapKeyPosesDS);
    for (auto & pt : globalMapKeyPosesDS.points) {
      kdtreeGlobalMap.nearestKSearch(
        pt, 1, pointSearchIndGlobalMap,
        pointSearchSqDisGlobalMap);
      pt.intensity = cloudKeyPoses3D.points[pointSearchIndGlobalMap[0]].intensity;
    }

    // extract visualized and downsampled key frames
    for (int i = 0; i < (int)globalMapKeyPosesDS.size(); ++i) {
      if (pointDistance(
          globalMapKeyPosesDS.points[i],
          cloudKeyPoses3D.back()) > globalMapVisualizationSearchRadius)
      {
        continue;
      }
      int thisKeyInd = (int)globalMapKeyPosesDS.points[i].intensity;
      *globalMapKeyFrames += transformPointCloud(
        cornerCloudKeyFrames[thisKeyInd],
        cloudKeyPoses6D.points[thisKeyInd]);
      *globalMapKeyFrames += transformPointCloud(
        surfCloudKeyFrames[thisKeyInd],
        cloudKeyPoses6D.points[thisKeyInd]);
    }
    // downsample visualized points
    // for global map visualization
    pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames;
    downSizeFilterGlobalMapKeyFrames.setLeafSize(
      globalMapVisualizationLeafSize,
      globalMapVisualizationLeafSize,
      globalMapVisualizationLeafSize);   // for global map visualization
    downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
    downSizeFilterGlobalMapKeyFrames.filter(globalMapKeyFramesDS);
    publishCloud(
      pubLaserCloudSurround, globalMapKeyFramesDS, timeLaserInfoStamp,
      odometryFrame);
  }

  void updateInitialGuess()
  {
    // save current transformation before any processing
    incrementalOdometryAffineFront = trans2Affine3d(posevec);

    static Eigen::Affine3d lastImuTransformation;

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
    static bool lastImuPreTransAvailable = false;
    static Eigen::Affine3d lastImuPreTransformation;
    if (cloudInfo.odomAvailable) {
      const Eigen::Affine3d back = poseToAffine(cloudInfo.initial_pose);
      if (!lastImuPreTransAvailable) {
        lastImuPreTransformation = back;
        lastImuPreTransAvailable = true;
      } else {
        const Eigen::Affine3d incre = lastImuPreTransformation.inverse() * back;
        const Eigen::Affine3d tobe = trans2Affine3d(posevec);
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

      const Eigen::Affine3d tobe = trans2Affine3d(posevec);
      posevec = getPoseVec(tobe * incre);

      // save imu before return;
      lastImuTransformation = makeAffine(rpy, Eigen::Vector3d::Zero());
      return;
    }
  }

  void extractNearby()
  {
    pcl::PointCloud<PointType>::Ptr surroundingKeyPoses(new
      pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS(new
      pcl::PointCloud<PointType>());
    std::vector<int> indices;
    std::vector<float> pointSearchSqDis;

    // extract all the nearby key poses and downsample them
    kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D.makeShared()); // create kd-tree
    kdtreeSurroundingKeyPoses->radiusSearch(
      cloudKeyPoses3D.back(),
      (double)surroundingKeyframeSearchRadius, indices, pointSearchSqDis);
    for (int i = 0; i < (int)indices.size(); ++i) {
      int id = indices[i];
      surroundingKeyPoses->push_back(cloudKeyPoses3D.points[id]);
    }

    downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
    downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS);
    for (auto & pt : surroundingKeyPosesDS->points) {
      kdtreeSurroundingKeyPoses->nearestKSearch(
        pt, 1, indices,
        pointSearchSqDis);
      pt.intensity = cloudKeyPoses3D.points[indices[0]].intensity;
    }

    // also extract some latest key frames in case the robot rotates in one position
    int numPoses = cloudKeyPoses3D.size();
    for (int i = numPoses - 1; i >= 0; --i) {
      if (timeLaserInfoCur - cloudKeyPoses6D.points[i].time < 10.0) {
        surroundingKeyPosesDS->push_back(cloudKeyPoses3D.points[i]);
      } else {
        break;
      }
    }

    extractCloud(surroundingKeyPosesDS);
  }

  void extractCloud(const pcl::PointCloud<PointType>::Ptr & cloudToExtract)
  {
    // fuse the map
    pcl::PointCloud<PointType>::Ptr corner(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr surface(new pcl::PointCloud<PointType>());

    for (int i = 0; i < (int)cloudToExtract->size(); ++i) {
      if (pointDistance(
          cloudToExtract->points[i],
          cloudKeyPoses3D.back()) > surroundingKeyframeSearchRadius)
      {
        continue;
      }

      int thisKeyInd = (int)cloudToExtract->points[i].intensity;
      if (laserCloudMapContainer.find(thisKeyInd) != laserCloudMapContainer.end()) {
        // transformed cloud available
        *corner += laserCloudMapContainer[thisKeyInd].first;
        *surface += laserCloudMapContainer[thisKeyInd].second;
        continue;
      }
      // transformed cloud not available
      pcl::PointCloud<PointType> c = transformPointCloud(
        cornerCloudKeyFrames[thisKeyInd], cloudKeyPoses6D.points[thisKeyInd]);
      pcl::PointCloud<PointType> s = transformPointCloud(
        surfCloudKeyFrames[thisKeyInd], cloudKeyPoses6D.points[thisKeyInd]);
      *corner += c;
      *surface += s;
      laserCloudMapContainer[thisKeyInd] = std::make_pair(c, s);
    }

    // Downsample the surrounding corner key frames (or map)
    downSizeFilterCorner.setInputCloud(corner);
    downSizeFilterCorner.filter(*laserCloudCornerFromMapDS);
    laserCloudCornerFromMapDSNum = laserCloudCornerFromMapDS->size();
    // Downsample the surrounding surf key frames (or map)
    downSizeFilterSurf.setInputCloud(surface);
    downSizeFilterSurf.filter(*laserCloudSurfFromMapDS);
    laserCloudSurfFromMapDSNum = laserCloudSurfFromMapDS->size();

    // clear map cache if too large
    if (laserCloudMapContainer.size() > 1000) {
      laserCloudMapContainer.clear();
    }
  }

  void extractSurroundingKeyFrames()
  {
    if (cloudKeyPoses3D.points.empty()) {
      return;
    }

    extractNearby();
  }

  void optimization(
    const pcl::PointCloud<PointType> & laserCloudCornerLastDS,
    const pcl::PointCloud<PointType> & laserCloudSurfLastDS)
  {
    const Eigen::Affine3d transPointAssociateToMap = trans2Affine3d(posevec);

    // corner optimization
    #pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < laserCloudCornerLastDS.size(); i++) {
      std::vector<int> indices;
      std::vector<float> pointSearchSqDis;

      const PointType pointOri = laserCloudCornerLastDS.points[i];
      const PointType pointSel = pointAssociateToMap(transPointAssociateToMap, pointOri);
      kdtreeCornerFromMap.nearestKSearch(pointSel, 5, indices, pointSearchSqDis);

      if (pointSearchSqDis[4] < 1.0) {
        Eigen::Vector3f c = Eigen::Vector3f::Zero();
        for (int j = 0; j < 5; j++) {
          c += laserCloudCornerFromMapDS->points[indices[j]].getVector3fMap();
        }
        c /= 5.0;

        Eigen::Matrix3f sa = Eigen::Matrix3f::Zero();

        for (int j = 0; j < 5; j++) {
          const Eigen::Vector3f x =
            laserCloudCornerFromMapDS->points[indices[j]].getVector3fMap();
          const Eigen::Vector3f a = x - c;
          sa += a * a.transpose();
        }

        sa = sa / 5.0;

        cv::Mat matA1(3, 3, CV_32F, cv::Scalar::all(0));
        cv::eigen2cv(sa, matA1);
        cv::Mat matD1(1, 3, CV_32F, cv::Scalar::all(0));
        cv::Mat matV1(3, 3, CV_32F, cv::Scalar::all(0));
        cv::eigen(matA1, matD1, matV1);
        Eigen::Matrix3f v1;
        cv::cv2eigen(matV1, v1);

        if (matD1.at<float>(0, 0) > 3 * matD1.at<float>(0, 1)) {
          const Eigen::Vector3f p0 = pointSel.getVector3fMap();
          const Eigen::Vector3f p1 = c + 0.1 * v1.row(0).transpose();
          const Eigen::Vector3f p2 = c - 0.1 * v1.row(0).transpose();

          const Eigen::Vector3f d01 = p0 - p1;
          const Eigen::Vector3f d02 = p0 - p2;
          const Eigen::Vector3f d12 = p1 - p2;

          // const Eigen::Vector3f d012(d01(0) * d02(1) - d02(0) * d01(1),
          //                            d01(0) * d02(2) - d02(0) * d01(2),
          //                            d01(1) * d02(2) - d02(1) * d01(2));
          const Eigen::Vector3f cross(d01(1) * d02(2) - d01(2) * d02(1),
            d01(2) * d02(0) - d01(0) * d02(2),
            d01(0) * d02(1) - d01(1) * d02(0));

          const float a012 = cross.norm();

          const float l12 = d12.norm();

          // possible bag. maybe the commented one is correct
          // float la = (d12(1) * cross(2) - cross(2) * d12(1)) / a012 / l12;
          // float lb = (d12(2) * cross(0) - cross(0) * d12(2)) / a012 / l12;
          // float lc = (d12(0) * cross(1) - cross(1) * d12(0)) / a012 / l12;

          float la = (d12(1) * cross(2) - d12(2) * cross(1)) / a012 / l12;
          float lb = (d12(2) * cross(0) - d12(0) * cross(2)) / a012 / l12;
          float lc = (d12(0) * cross(1) - d12(1) * cross(0)) / a012 / l12;

          float ld2 = a012 / l12;

          float s = 1 - 0.9 * fabs(ld2);

          PointType coeff;
          coeff.x = s * la;
          coeff.y = s * lb;
          coeff.z = s * lc;
          coeff.intensity = s * ld2;

          if (s > 0.1) {
            laserCloudOriCornerVec[i] = pointOri;
            coeffSelCornerVec[i] = coeff;
            laserCloudOriCornerFlag[i] = true;
          }
        }
      }
    }

    // surface optimization
    #pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < laserCloudSurfLastDS.size(); i++) {
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

      const Eigen::Vector3d p = pointSel.getVector3fMap().cast<double>();
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

    // combine corner coeffs
    for (int i = 0; i < laserCloudCornerLastDS.size(); ++i) {
      if (laserCloudOriCornerFlag[i]) {
        laserCloudOri->push_back(laserCloudOriCornerVec[i]);
        coeffSel->push_back(coeffSelCornerVec[i]);
      }
    }
    // combine surf coeffs
    for (int i = 0; i < laserCloudSurfLastDS.size(); ++i) {
      if (laserCloudOriSurfFlag[i]) {
        laserCloudOri->push_back(laserCloudOriSurfVec[i]);
        coeffSel->push_back(coeffSelSurfVec[i]);
      }
    }
    // reset flag for next iteration
    std::fill(
      laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(),
      false);
    std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);
  }

  bool LMOptimization(int iterCount)
  {
    // This optimization is from the original loam_velodyne by Ji Zhang, need to cope with coordinate transformation
    // lidar <- camera      ---     camera <- lidar
    // x = z                ---     x = y
    // y = x                ---     y = z
    // z = y                ---     z = x
    // roll = yaw           ---     roll = pitch
    // pitch = roll         ---     pitch = yaw
    // yaw = pitch          ---     yaw = roll

    // lidar -> camera
    int laserCloudSelNum = laserCloudOri->size();
    if (laserCloudSelNum < 50) {
      return false;
    }

    cv::Mat matA(laserCloudSelNum, 6, CV_32F, cv::Scalar::all(0));
    cv::Mat matB(laserCloudSelNum, 1, CV_32F, cv::Scalar::all(0));

    for (int i = 0; i < laserCloudSelNum; i++) {
      // lidar -> camera
      const float intensity = coeffSel->points[i].intensity;

      // in camera

      const Eigen::Vector3f point_ori(laserCloudOri->points[i].y,
        laserCloudOri->points[i].z,
        laserCloudOri->points[i].x);

      const Eigen::Vector3f coeff_vec(coeffSel->points[i].y,
        coeffSel->points[i].z,
        coeffSel->points[i].x);

      const Eigen::Matrix3f MX = dRdx(posevec(0), posevec(2), posevec(1));
      const float arx = (MX * point_ori).dot(coeff_vec);

      const Eigen::Matrix3f MY = dRdy(posevec(0), posevec(2), posevec(1));
      const float ary = (MY * point_ori).dot(coeff_vec);

      const Eigen::Matrix3f MZ = dRdz(posevec(0), posevec(2), posevec(1));
      const float arz = (MZ * point_ori).dot(coeff_vec);

      // lidar -> camera
      matA.at<float>(i, 0) = arz;
      matA.at<float>(i, 1) = arx;
      matA.at<float>(i, 2) = ary;
      matA.at<float>(i, 3) = coeffSel->points[i].x;
      matA.at<float>(i, 4) = coeffSel->points[i].y;
      matA.at<float>(i, 5) = coeffSel->points[i].z;
      matB.at<float>(i, 0) = -intensity;
    }

    cv::Mat matAt(6, laserCloudSelNum, CV_32F, cv::Scalar::all(0));
    cv::transpose(matA, matAt);
    const cv::Mat matAtA = matAt * matA;
    const cv::Mat matAtB = matAt * matB;

    cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));
    cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

    cv::Mat matP = cv::Mat(6, 6, CV_32F, cv::Scalar::all(0));
    if (iterCount == 0) {

      cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
      cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
      cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));

      cv::eigen(matAtA, matE, matV);

      isDegenerate = false;
      float eignThre[6] = {100, 100, 100, 100, 100, 100};
      for (int i = 5; i >= 0; i--) {
        if (matE.at<float>(0, i) < eignThre[i]) {
          for (int j = 0; j < 6; j++) {
            matV2.at<float>(i, j) = 0;
          }
          isDegenerate = true;
        } else {
          break;
        }
      }
      matP = matV.inv() * matV2;
    }

    if (isDegenerate) {
      cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
      matX.copyTo(matX2);
      matX = matP * matX2;
    }

    posevec(0) += matX.at<float>(0, 0);
    posevec(1) += matX.at<float>(1, 0);
    posevec(2) += matX.at<float>(2, 0);
    posevec(3) += matX.at<float>(3, 0);
    posevec(4) += matX.at<float>(4, 0);
    posevec(5) += matX.at<float>(5, 0);

    float deltaR = sqrt(
      pow(pcl::rad2deg(matX.at<float>(0, 0)), 2) +
      pow(pcl::rad2deg(matX.at<float>(1, 0)), 2) +
      pow(pcl::rad2deg(matX.at<float>(2, 0)), 2));
    float deltaT = sqrt(
      pow(matX.at<float>(3, 0) * 100, 2) +
      pow(matX.at<float>(4, 0) * 100, 2) +
      pow(matX.at<float>(5, 0) * 100, 2));

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

    if (laserCloudCornerLastDS.size() > edgeFeatureMinValidNum &&
      laserCloudSurfLastDS.size() > surfFeatureMinValidNum)
    {
      kdtreeCornerFromMap.setInputCloud(laserCloudCornerFromMapDS);
      kdtreeSurfFromMap.setInputCloud(laserCloudSurfFromMapDS);

      for (int iterCount = 0; iterCount < 30; iterCount++) {
        laserCloudOri->clear();
        coeffSel->clear();

        optimization(laserCloudCornerLastDS, laserCloudSurfLastDS);

        if (LMOptimization(iterCount)) {
          break;
        }
      }

      transformUpdate();
    } else {
      ROS_WARN(
        "Not enough features! Only %d edge and %d planar features available.",
        laserCloudCornerLastDS.size(), laserCloudSurfLastDS.size());
    }
  }

  void transformUpdate()
  {
    if (cloudInfo.imuAvailable) {
      if (std::abs(cloudInfo.initialIMU.y) < 1.4) {
        double imuWeight = imuRPYWeight;
        tf::Quaternion imuQuaternion;
        tf::Quaternion transformQuaternion;

        // slerp roll
        transformQuaternion.setRPY(posevec(0), 0, 0);
        imuQuaternion.setRPY(cloudInfo.initialIMU.x, 0, 0);
        posevec(0) = getRPY(interpolate(transformQuaternion, imuQuaternion, imuWeight))(0);

        // slerp pitch
        transformQuaternion.setRPY(0, posevec(1), 0);
        imuQuaternion.setRPY(0, cloudInfo.initialIMU.y, 0);
        posevec(1) = getRPY(interpolate(transformQuaternion, imuQuaternion, imuWeight))(1);
      }
    }

    posevec(0) = constraintTransformation(posevec(0), rotation_tolerance);
    posevec(1) = constraintTransformation(posevec(1), rotation_tolerance);
    posevec(5) = constraintTransformation(posevec(5), z_tolerance);

    incrementalOdometryAffineBack = trans2Affine3d(posevec);
  }

  bool saveFrame()
  {
    if (cloudKeyPoses3D.points.empty()) {
      return true;
    }

    Eigen::Affine3d transStart = pclPointToAffine3d(cloudKeyPoses6D.back());
    Eigen::Affine3d transFinal = getTransformation(posevec);
    Eigen::Affine3d transBetween = transStart.inverse() * transFinal;
    double x, y, z, roll, pitch, yaw;
    pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw);

    if (abs(roll) < surroundingkeyframeAddingAngleThreshold &&
      abs(pitch) < surroundingkeyframeAddingAngleThreshold &&
      abs(yaw) < surroundingkeyframeAddingAngleThreshold &&
      sqrt(x * x + y * y + z * z) < surroundingkeyframeAddingDistThreshold)
    {
      return false;
    }

    return true;
  }

  void addOdomFactor()
  {
    if (cloudKeyPoses3D.points.empty()) {
      // rad*rad, meter*meter
      const Eigen::MatrixXd v = (Vector(6) << 1e-2, 1e-2, M_PI * M_PI, 1e8, 1e8, 1e8).finished();
      const noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Variances(v);
      gtSAMgraph.add(PriorFactor<Pose3>(0, trans2gtsamPose(posevec), priorNoise));
      initialEstimate.insert(0, trans2gtsamPose(posevec));
    } else {
      const Eigen::MatrixXd v = (Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished();
      const noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Variances(v);
      gtsam::Pose3 poseFrom = pclPointTogtsamPose3(cloudKeyPoses6D.points.back());
      gtsam::Pose3 poseTo = trans2gtsamPose(posevec);
      gtSAMgraph.add(
        BetweenFactor<Pose3>(
          cloudKeyPoses3D.size() - 1,
          cloudKeyPoses3D.size(), poseFrom.between(poseTo), odometryNoise));
      initialEstimate.insert(cloudKeyPoses3D.size(), poseTo);
    }
  }

  void addGPSFactor()
  {
    if (gpsQueue.empty()) {
      return;
    }

    // wait for system initialized and settles down
    if (cloudKeyPoses3D.points.empty()) {
      return;
    } else {
      if (pointDistance(cloudKeyPoses3D.front(), cloudKeyPoses3D.back()) < 5.0) {
        return;
      }
    }

    // pose covariance small, no need to correct
    if (poseCovariance(3, 3) < poseCovThreshold &&
      poseCovariance(4, 4) < poseCovThreshold)
    {
      return;
    }

    // last gps position
    static PointType lastGPSPoint;

    while (!gpsQueue.empty()) {
      if (gpsQueue.front().header.stamp.toSec() < timeLaserInfoCur - 0.2) {
        // message too old
        gpsQueue.pop_front();
        continue;
      }

      if (gpsQueue.front().header.stamp.toSec() > timeLaserInfoCur + 0.2) {
        // message too new
        return;
      }

      nav_msgs::Odometry thisGPS = gpsQueue.front();
      gpsQueue.pop_front();

      // GPS too noisy, skip
      float noise_x = thisGPS.pose.covariance[0];
      float noise_y = thisGPS.pose.covariance[7];
      float noise_z = thisGPS.pose.covariance[14];
      if (noise_x > gpsCovThreshold || noise_y > gpsCovThreshold) {
        continue;
      }

      float gps_x = thisGPS.pose.pose.position.x;
      float gps_y = thisGPS.pose.pose.position.y;
      float gps_z = thisGPS.pose.pose.position.z;
      if (!useGpsElevation) {
        gps_z = posevec(5);
        noise_z = 0.01;
      }

      // GPS not properly initialized (0,0,0)
      if (abs(gps_x) < 1e-6 && abs(gps_y) < 1e-6) {
        continue;
      }

      // Add GPS every a few meters
      PointType curGPSPoint;
      curGPSPoint.x = gps_x;
      curGPSPoint.y = gps_y;
      curGPSPoint.z = gps_z;
      if (pointDistance(curGPSPoint, lastGPSPoint) < 5.0) {
        continue;
      } else {
        lastGPSPoint = curGPSPoint;
      }

      const gtsam::Vector3 Vector(noise_x, noise_y, noise_z);
      const auto gps_noise = noiseModel::Diagonal::Variances(Vector.cwiseMax(1.0f));
      gtsam::GPSFactor gps_factor(cloudKeyPoses3D.size(), gtsam::Point3(
          gps_x, gps_y,
          gps_z), gps_noise);
      gtSAMgraph.add(gps_factor);

      aLoopIsClosed = true;
      return;
    }
  }

  void saveKeyFramesAndFactor(
    const pcl::PointCloud<PointType> & laserCloudCornerLastDS,
    const pcl::PointCloud<PointType> & laserCloudSurfLastDS)
  {
    if (!saveFrame()) {
      return;
    }

    // odom factor
    addOdomFactor();

    // gps factor
    addGPSFactor();

    // std::cout << "****************************************************" << std::endl;
    // gtSAMgraph.print("GTSAM Graph:\n");

    // update iSAM
    isam->update(gtSAMgraph, initialEstimate);
    isam->update();

    if (aLoopIsClosed) {
      isam->update();
      isam->update();
      isam->update();
      isam->update();
      isam->update();
    }

    gtSAMgraph.resize(0);
    initialEstimate.clear();

    //save key poses
    PointType thisPose3D;
    PointTypePose thisPose6D;
    Pose3 latestEstimate;

    isamCurrentEstimate = isam->calculateEstimate();
    latestEstimate = isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size() - 1);
    // std::cout << "****************************************************" << std::endl;
    // isamCurrentEstimate.print("Current estimate: ");

    thisPose3D.x = latestEstimate.translation().x();
    thisPose3D.y = latestEstimate.translation().y();
    thisPose3D.z = latestEstimate.translation().z();
    thisPose3D.intensity = cloudKeyPoses3D.size(); // this can be used as index
    cloudKeyPoses3D.push_back(thisPose3D);

    thisPose6D.x = thisPose3D.x;
    thisPose6D.y = thisPose3D.y;
    thisPose6D.z = thisPose3D.z;
    thisPose6D.intensity = thisPose3D.intensity;  // this can be used as index
    thisPose6D.roll = latestEstimate.rotation().roll();
    thisPose6D.pitch = latestEstimate.rotation().pitch();
    thisPose6D.yaw = latestEstimate.rotation().yaw();
    thisPose6D.time = timeLaserInfoCur;
    cloudKeyPoses6D.push_back(thisPose6D);

    // std::cout << "****************************************************" << std::endl;
    // std::cout << "Pose covariance:" << std::endl;
    // std::cout << isam->marginalCovariance(isamCurrentEstimate.size()-1) << std::endl << std::endl;
    poseCovariance = isam->marginalCovariance(isamCurrentEstimate.size() - 1);

    // save updated transform
    posevec = getPoseVec(latestEstimate);

    // save key frame cloud
    cornerCloudKeyFrames.push_back(laserCloudCornerLastDS);
    surfCloudKeyFrames.push_back(laserCloudSurfLastDS);

    // save path for visualization
    updatePath(thisPose6D);
  }

  void correctPoses()
  {
    if (cloudKeyPoses3D.points.empty()) {
      return;
    }

    if (aLoopIsClosed) {
      // clear map cache
      laserCloudMapContainer.clear();
      // clear path
      globalPath.poses.clear();
      // update key poses
      int numPoses = isamCurrentEstimate.size();
      for (int i = 0; i < numPoses; ++i) {
        const auto t = isamCurrentEstimate.at<Pose3>(i).translation();
        cloudKeyPoses3D.points[i].x = t.x();
        cloudKeyPoses3D.points[i].y = t.y();
        cloudKeyPoses3D.points[i].z = t.z();

        const auto p = cloudKeyPoses3D.points[i];
        cloudKeyPoses6D.points[i].x = p.x;
        cloudKeyPoses6D.points[i].y = p.y;
        cloudKeyPoses6D.points[i].z = p.z;
        const auto r = isamCurrentEstimate.at<Pose3>(i).rotation();
        cloudKeyPoses6D.points[i].roll = r.roll();
        cloudKeyPoses6D.points[i].pitch = r.pitch();
        cloudKeyPoses6D.points[i].yaw = r.yaw();

        updatePath(cloudKeyPoses6D.points[i]);
      }

      aLoopIsClosed = false;
    }
  }

  void updatePath(const PointTypePose & pose_in)
  {
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header.stamp = ros::Time().fromSec(pose_in.time);
    pose_stamped.header.frame_id = odometryFrame;
    pose_stamped.pose.position.x = pose_in.x;
    pose_stamped.pose.position.y = pose_in.y;
    pose_stamped.pose.position.z = pose_in.z;
    pose_stamped.pose.orientation = \
      tf::createQuaternionMsgFromRollPitchYaw(pose_in.roll, pose_in.pitch, pose_in.yaw);

    globalPath.poses.push_back(pose_stamped);
  }

  void publishOdometry()
  {
    // Publish odometry for ROS (global)
    nav_msgs::Odometry laserOdometryROS;
    laserOdometryROS.header.stamp = timeLaserInfoStamp;
    laserOdometryROS.header.frame_id = odometryFrame;
    laserOdometryROS.child_frame_id = "odom_mapping";
    laserOdometryROS.pose.pose = makePose(posevec);
    // geometry_msgs/Quaternion
    pubLaserOdometryGlobal.publish(laserOdometryROS);

    // Publish TF
    static tf::TransformBroadcaster br;
    tf::Transform t_odom_to_lidar = makeTransform(posevec);
    tf::StampedTransform trans_odom_to_lidar = tf::StampedTransform(
      t_odom_to_lidar, timeLaserInfoStamp, odometryFrame, "lidar_link");
    br.sendTransform(trans_odom_to_lidar);

    // Publish odometry for ROS (incremental)
    static bool lastIncreOdomPubFlag = false;
    static nav_msgs::Odometry laserOdomIncremental; // incremental odometry msg
    static Eigen::Affine3d increOdomAffine; // incremental odometry in affine
    if (!lastIncreOdomPubFlag) {
      lastIncreOdomPubFlag = true;
      laserOdomIncremental = laserOdometryROS;
      increOdomAffine = trans2Affine3d(posevec);
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
      laserOdomIncremental.header.stamp = timeLaserInfoStamp;
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
    const pcl::PointCloud<PointType> & laserCloudSurfLastDS)
  {
    if (cloudKeyPoses3D.points.empty()) {
      return;
    }
    // publish key poses
    publishCloud(pubKeyPoses, cloudKeyPoses3D, timeLaserInfoStamp, odometryFrame);
    // Publish surrounding key frames
    publishCloud(
      pubRecentKeyFrames, *laserCloudSurfFromMapDS, timeLaserInfoStamp,
      odometryFrame);
    // publish registered key frame
    if (pubRecentKeyFrame.getNumSubscribers() != 0) {
      pcl::PointCloud<PointType> cloudOut;
      PointTypePose thisPose6D = trans2PointTypePose(posevec);
      cloudOut += transformPointCloud(laserCloudCornerLastDS, thisPose6D);
      cloudOut += transformPointCloud(laserCloudSurfLastDS, thisPose6D);
      publishCloud(pubRecentKeyFrame, cloudOut, timeLaserInfoStamp, odometryFrame);
    }
    // publish registered high-res raw cloud
    if (pubCloudRegisteredRaw.getNumSubscribers() != 0) {
      const pcl::PointCloud<PointType> cloudOut =
        getPointCloud<PointType>(cloudInfo.cloud_deskewed);
      const PointTypePose thisPose6D = trans2PointTypePose(posevec);
      publishCloud(
        pubCloudRegisteredRaw, transformPointCloud(cloudOut, thisPose6D),
        timeLaserInfoStamp, odometryFrame);
    }
    // publish path
    if (pubPath.getNumSubscribers() != 0) {
      globalPath.header.stamp = timeLaserInfoStamp;
      globalPath.header.frame_id = odometryFrame;
      pubPath.publish(globalPath);
    }
  }
};


int main(int argc, char ** argv)
{
  ros::init(argc, argv, "lio_sam");

  mapOptimization MO;

  ROS_INFO("\033[1;32m----> Map Optimization Started.\033[0m");

  std::thread visualizeMapThread(&mapOptimization::visualizeGlobalMapThread,
    &MO);

  ros::spin();

  visualizeMapThread.join();

  return 0;
}
