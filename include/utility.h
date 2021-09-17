#pragma once
#ifndef _UTILITY_LIDAR_ODOMETRY_H_
#define _UTILITY_LIDAR_ODOMETRY_H_

#include <ros/ros.h>

#include <std_msgs/Header.h>
#include <std_msgs/Float64MultiArray.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/NavSatFix.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <opencv/cv.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl_conversions/pcl_conversions.h>

#include <tf/LinearMath/Quaternion.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

#include <eigen_conversions/eigen_msg.h>

#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>
#include <deque>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cfloat>
#include <iterator>
#include <sstream>
#include <string>
#include <limits>
#include <iomanip>
#include <array>
#include <thread>
#include <tuple>
#include <mutex>

typedef pcl::PointXYZI PointType;
typedef Eigen::Matrix < double, -1, -1, Eigen::RowMajor > RowMajorMatrixXd;
enum class SensorType {VELODYNE, OUSTER};

class ParamServer {
public:
  ros::NodeHandle nh;

  std::string robot_id;

  //Topics
  std::string pointCloudTopic;
  std::string imuTopic;
  std::string odomTopic;
  std::string gpsTopic;

  //Frames
  std::string lidarFrame;
  std::string baselinkFrame;
  std::string odometryFrame;
  std::string mapFrame;

  // GPS Settings
  bool useImuHeadingInitialization;
  bool useGpsElevation;
  float gpsCovThreshold;
  float poseCovThreshold;

  // Save pcd
  bool savePCD;
  std::string savePCDDirectory;

  // Lidar Sensor Configuration
  SensorType sensor;
  int N_SCAN;
  int Horizon_SCAN;
  int downsampleRate;
  float lidarMinRange;
  float lidarMaxRange;

  // IMU
  float imuAccNoise;
  float imuGyrNoise;
  float imuAccBiasN;
  float imuGyrBiasN;
  float imuGravity;
  float imuRPYWeight;
  std::vector < double > extTransV;
  Eigen::Vector3d extTrans;

  // LOAM
  float edgeThreshold;
  float surfThreshold;
  int edgeFeatureMinValidNum;
  int surfFeatureMinValidNum;

  // voxel filter paprams
  float odometrySurfLeafSize;
  float mappingCornerLeafSize;
  float mappingSurfLeafSize;

  float z_tolerance;
  float rotation_tolerance;

  // CPU Params
  int numberOfCores;
  double mappingProcessInterval;

  // Surrounding map
  float surroundingkeyframeAddingDistThreshold;
  float surroundingkeyframeAddingAngleThreshold;
  float surroundingKeyframeDensity;
  float surroundingKeyframeSearchRadius;

  // Loop closure
  bool loopClosureEnableFlag;
  float loopClosureFrequency;
  int surroundingKeyframeSize;
  float historyKeyframeSearchRadius;
  float historyKeyframeSearchTimeDiff;
  int historyKeyframeSearchNum;
  float historyKeyframeFitnessScore;

  // global map visualization radius
  float globalMapVisualizationSearchRadius;
  float globalMapVisualizationPoseDensity;
  float globalMapVisualizationLeafSize;

  ParamServer() {
    nh.param < std::string > ("/robot_id", robot_id, "roboat");

    nh.param < std::string > ("lio_sam/pointCloudTopic", pointCloudTopic,
    "points_raw");
    nh.param < std::string > ("lio_sam/imuTopic", imuTopic, "imu_correct");
    nh.param < std::string > ("lio_sam/odomTopic", odomTopic, "odometry/imu");
    nh.param < std::string > ("lio_sam/gpsTopic", gpsTopic, "odometry/gps");

    nh.param < std::string > ("lio_sam/lidarFrame", lidarFrame, "base_link");
    nh.param < std::string > ("lio_sam/baselinkFrame", baselinkFrame, "base_link");
    nh.param < std::string > ("lio_sam/odometryFrame", odometryFrame, "odom");
    nh.param < std::string > ("lio_sam/mapFrame", mapFrame, "map");

    nh.param < bool > ("lio_sam/useImuHeadingInitialization",
    useImuHeadingInitialization, false);
    nh.param < bool > ("lio_sam/useGpsElevation", useGpsElevation, false);
    nh.param < float > ("lio_sam/gpsCovThreshold", gpsCovThreshold, 2.0);
    nh.param < float > ("lio_sam/poseCovThreshold", poseCovThreshold, 25.0);

    nh.param < bool > ("lio_sam/savePCD", savePCD, false);
    nh.param < std::string > ("lio_sam/savePCDDirectory", savePCDDirectory,
    "/Downloads/LOAM/");

    std::string sensorStr;
    nh.param < std::string > ("lio_sam/sensor", sensorStr, "");
    if (sensorStr == "velodyne") {
      sensor = SensorType::VELODYNE;
    } else if (sensorStr == "ouster") {
      sensor = SensorType::OUSTER;
    } else {
      ROS_ERROR_STREAM(
        "Invalid sensor type (must be either 'velodyne' or 'ouster'): " << sensorStr);
      ros::shutdown();
    }

    nh.param < int > ("lio_sam/N_SCAN", N_SCAN, 16);
    nh.param < int > ("lio_sam/Horizon_SCAN", Horizon_SCAN, 1800);
    nh.param < int > ("lio_sam/downsampleRate", downsampleRate, 1);
    nh.param < float > ("lio_sam/lidarMinRange", lidarMinRange, 1.0);
    nh.param < float > ("lio_sam/lidarMaxRange", lidarMaxRange, 1000.0);

    nh.param < float > ("lio_sam/imuAccNoise", imuAccNoise, 0.01);
    nh.param < float > ("lio_sam/imuGyrNoise", imuGyrNoise, 0.001);
    nh.param < float > ("lio_sam/imuAccBiasN", imuAccBiasN, 0.0002);
    nh.param < float > ("lio_sam/imuGyrBiasN", imuGyrBiasN, 0.00003);
    nh.param < float > ("lio_sam/imuGravity", imuGravity, 9.80511);
    nh.param < float > ("lio_sam/imuRPYWeight", imuRPYWeight, 0.01);
    nh.param < std::vector <
    double >> ("lio_sam/extrinsicTrans", extTransV, std::vector < double > ());
    extTrans = Eigen::Map < const RowMajorMatrixXd > (extTransV.data(), 3, 1);

    nh.param < float > ("lio_sam/edgeThreshold", edgeThreshold, 0.1);
    nh.param < float > ("lio_sam/surfThreshold", surfThreshold, 0.1);
    nh.param < int > ("lio_sam/edgeFeatureMinValidNum", edgeFeatureMinValidNum, 10);
    nh.param < int > ("lio_sam/surfFeatureMinValidNum", surfFeatureMinValidNum, 100);

    nh.param < float > ("lio_sam/odometrySurfLeafSize", odometrySurfLeafSize, 0.2);
    nh.param < float > ("lio_sam/mappingCornerLeafSize", mappingCornerLeafSize, 0.2);
    nh.param < float > ("lio_sam/mappingSurfLeafSize", mappingSurfLeafSize, 0.2);

    nh.param < float > ("lio_sam/z_tolerance", z_tolerance, FLT_MAX);
    nh.param < float > ("lio_sam/rotation_tolerance", rotation_tolerance, FLT_MAX);

    nh.param < int > ("lio_sam/numberOfCores", numberOfCores, 2);
    nh.param < double > ("lio_sam/mappingProcessInterval", mappingProcessInterval,
    0.15);

    nh.param < float > ("lio_sam/surroundingkeyframeAddingDistThreshold",
    surroundingkeyframeAddingDistThreshold, 1.0);
    nh.param < float > ("lio_sam/surroundingkeyframeAddingAngleThreshold",
    surroundingkeyframeAddingAngleThreshold, 0.2);
    nh.param < float > ("lio_sam/surroundingKeyframeDensity",
    surroundingKeyframeDensity, 1.0);
    nh.param < float > ("lio_sam/surroundingKeyframeSearchRadius",
    surroundingKeyframeSearchRadius, 50.0);

    nh.param < bool > ("lio_sam/loopClosureEnableFlag", loopClosureEnableFlag, false);
    nh.param < float > ("lio_sam/loopClosureFrequency", loopClosureFrequency, 1.0);
    nh.param < int > ("lio_sam/surroundingKeyframeSize", surroundingKeyframeSize, 50);
    nh.param < float > ("lio_sam/historyKeyframeSearchRadius",
    historyKeyframeSearchRadius, 10.0);
    nh.param < float > ("lio_sam/historyKeyframeSearchTimeDiff",
    historyKeyframeSearchTimeDiff, 30.0);
    nh.param < int > ("lio_sam/historyKeyframeSearchNum", historyKeyframeSearchNum,
    25);
    nh.param < float > ("lio_sam/historyKeyframeFitnessScore",
    historyKeyframeFitnessScore, 0.3);

    nh.param < float > ("lio_sam/globalMapVisualizationSearchRadius",
    globalMapVisualizationSearchRadius, 1e3);
    nh.param < float > ("lio_sam/globalMapVisualizationPoseDensity",
    globalMapVisualizationPoseDensity, 10.0);
    nh.param < float > ("lio_sam/globalMapVisualizationLeafSize",
    globalMapVisualizationLeafSize, 1.0);

    usleep(100);
  }
};

class IMUConverter {
public:
  IMUConverter() {
    std::vector < double > extRotV;
    std::vector < double > extRPYV;
    nh.param < std::vector < double >> ("lio_sam/extrinsicRot", extRotV, std::vector < double > ());
    nh.param < std::vector < double >> ("lio_sam/extrinsicRPY", extRPYV, std::vector < double > ());
    extRot = Eigen::Map < const RowMajorMatrixXd > (extRotV.data(), 3, 3);
    Eigen::Matrix3d extRPY = Eigen::Map < const RowMajorMatrixXd > (extRPYV.data(), 3, 3);
    extQRPY = Eigen::Quaterniond(extRPY);
  }

  sensor_msgs::Imu imuConverter(const sensor_msgs::Imu & imu_in)
  {
    sensor_msgs::Imu imu_out = imu_in;
    // rotate acceleration
    Eigen::Vector3d acc(
      imu_in.linear_acceleration.x,
      imu_in.linear_acceleration.y,
      imu_in.linear_acceleration.z);
    acc = extRot * acc;
    imu_out.linear_acceleration.x = acc.x();
    imu_out.linear_acceleration.y = acc.y();
    imu_out.linear_acceleration.z = acc.z();
    // rotate gyroscope
    Eigen::Vector3d gyr(
      imu_in.angular_velocity.x,
      imu_in.angular_velocity.y,
      imu_in.angular_velocity.z);
    gyr = extRot * gyr;
    imu_out.angular_velocity.x = gyr.x();
    imu_out.angular_velocity.y = gyr.y();
    imu_out.angular_velocity.z = gyr.z();
    // rotate roll pitch yaw
    Eigen::Quaterniond q_from(
      imu_in.orientation.w, imu_in.orientation.x,
      imu_in.orientation.y, imu_in.orientation.z);
    Eigen::Quaterniond q_final = q_from * extQRPY;
    imu_out.orientation.x = q_final.x();
    imu_out.orientation.y = q_final.y();
    imu_out.orientation.z = q_final.z();
    imu_out.orientation.w = q_final.w();

    if (sqrt(
        q_final.x() * q_final.x() + q_final.y() * q_final.y() + q_final.z() *
        q_final.z() + q_final.w() * q_final.w()) < 0.1)
    {
      ROS_ERROR("Invalid quaternion, please use a 9-axis IMU!");
      ros::shutdown();
    }

    return imu_out;
  }

private:
  ros::NodeHandle nh;
  Eigen::Matrix3d extRot;
  Eigen::Quaterniond extQRPY;
};

sensor_msgs::PointCloud2 publishCloud(
  ros::Publisher * thisPub,
  pcl::PointCloud < PointType > & thisCloud, ros::Time thisStamp,
  std::string thisFrame)
{
  sensor_msgs::PointCloud2 tempCloud;
  pcl::toROSMsg(thisCloud, tempCloud);
  tempCloud.header.stamp = thisStamp;
  tempCloud.header.frame_id = thisFrame;
  if (thisPub->getNumSubscribers() != 0) {
    thisPub->publish(tempCloud);
  }
  return tempCloud;
}

template < typename T >
double ROS_TIME(T msg)
{
  return msg->header.stamp.toSec();
}

Eigen::Vector3d pointToEigen(const geometry_msgs::Point & p)
{
  return Eigen::Vector3d(p.x, p.y, p.z);
}

Eigen::Vector3d vector3ToEigen(const geometry_msgs::Vector3 & p)
{
  return Eigen::Vector3d(p.x, p.y, p.z);
}

Eigen::Affine3d makeAffine(const Eigen::Vector3d & rpy, const Eigen::Vector3d & point)
{
  Eigen::Affine3d transform;
  pcl::getTransformation(point(0), point(1), point(2), rpy(0), rpy(1), rpy(2), transform);
  return transform;
}

Eigen::Affine3d makeAffine(
  const geometry_msgs::Vector3 & rpy,
  const geometry_msgs::Vector3 & point)
{
  Eigen::Affine3d transform;
  pcl::getTransformation(point.x, point.y, point.z, rpy.x, rpy.y, rpy.z, transform);
  return transform;
}

Eigen::Vector3d imuAngular2rosAngular(
  const geometry_msgs::Vector3 & angular_velocity)
{
  return Eigen::Vector3d(angular_velocity.x, angular_velocity.y, angular_velocity.z);
}

Eigen::Vector3d quaternionToRPY(const tf::Quaternion & orientation)
{
  Eigen::Vector3d rpy;
  tf::Matrix3x3(orientation).getRPY(rpy(0), rpy(1), rpy(2));
  return rpy;
}

Eigen::Vector3d quaternionToRPY(const geometry_msgs::Quaternion & orientation)
{
  tf::Quaternion quat;
  tf::quaternionMsgToTF(orientation, quat);
  return quaternionToRPY(quat);
}

Eigen::Affine3d poseToAffine(const geometry_msgs::Pose & pose)
{
  Eigen::Affine3d affine;
  tf::poseMsgToEigen(pose, affine);
  return affine;
}

geometry_msgs::Vector3 eigenToVector3(const Eigen::Vector3d & v)
{
  geometry_msgs::Vector3 p;
  p.x = v[0];
  p.y = v[1];
  p.z = v[2];
  return p;
}

geometry_msgs::Point eigenToPoint(const Eigen::Vector3d & v)
{
  geometry_msgs::Point p;
  p.x = v[0];
  p.y = v[1];
  p.z = v[2];
  return p;
}

geometry_msgs::Pose affineToPose(const Eigen::Affine3d & affine)
{
  geometry_msgs::Pose pose;
  tf::poseEigenToMsg(affine, pose);
  return pose;
}

template < typename PointType >
struct Points
{
  typedef std::vector < PointType, Eigen::aligned_allocator < PointType >> type;
};

template < typename PointType >
pcl::PointCloud < PointType > getPointCloud(const sensor_msgs::PointCloud2 & roscloud)
{
  pcl::PointCloud < PointType > pclcloud;
  pcl::fromROSMsg(roscloud, pclcloud);
  return pclcloud;
}

float pointDistance(PointType p1, PointType p2)
{
  return sqrt(
    (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) *
    (p1.z - p2.z));
}

double timeInSec(const std_msgs::Header & header)
{
  return header.stamp.toSec();
}

template < typename T >
void dropBefore(const double time, std::deque < T > & buffer)
{
  while (!buffer.empty()) {
    if (timeInSec(buffer.front().header) < time) {
      buffer.pop_front();
    } else {
      break;
    }
  }
}

#endif
