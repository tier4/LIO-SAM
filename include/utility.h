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
#include <gtsam/geometry/Pose3.h>

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
typedef Eigen::Matrix < double, 6, 1 > Vector6d;

sensor_msgs::PointCloud2 toRosMsg(const pcl::PointCloud < PointType > & pointcloud);

sensor_msgs::PointCloud2 publishCloud(
  const ros::Publisher & thisPub,
  const pcl::PointCloud < PointType > & thisCloud,
  const ros::Time thisStamp,
  const std::string thisFrame);

Eigen::Vector3d pointToEigen(const geometry_msgs::Point & p)
{
  return Eigen::Vector3d(p.x, p.y, p.z);
}

Eigen::Vector3d vector3ToEigen(const geometry_msgs::Vector3 & p)
{
  return Eigen::Vector3d(p.x, p.y, p.z);
}

PointType makePoint(const Eigen::Vector3d & point, const float intensity = 0.0);

Eigen::Affine3d makeAffine(
  const Eigen::Vector3d & rpy = Eigen::Vector3d::Zero(),
  const Eigen::Vector3d & point = Eigen::Vector3d::Zero());

Eigen::Affine3d makeAffine(
  const geometry_msgs::Vector3 & rpy,
  const geometry_msgs::Vector3 & point);

Eigen::Vector3d quaternionToRPY(const tf::Quaternion & orientation);

Eigen::Vector3d quaternionToRPY(const geometry_msgs::Quaternion & orientation);

Eigen::Quaterniond quaternionToEigen(const geometry_msgs::Quaternion & quat_msg);

geometry_msgs::Quaternion eigenToQuaternion(const Eigen::Quaterniond & quat_eigen);

Eigen::Affine3d poseToAffine(const geometry_msgs::Pose & pose);

geometry_msgs::Vector3 eigenToVector3(const Eigen::Vector3d & v);

std::tuple < Eigen::Vector3d, Eigen::Vector3d > getXYZRPY(const Eigen::Affine3d & affine);

geometry_msgs::Point eigenToPoint(const Eigen::Vector3d & v);

geometry_msgs::Pose makePose(
  const geometry_msgs::Quaternion & orientation,
  const geometry_msgs::Point & position);

geometry_msgs::Twist makeTwist(
  const geometry_msgs::Vector3 & angular,
  const geometry_msgs::Vector3 & linear);

geometry_msgs::Pose makePose(const gtsam::Pose3 & pose);

geometry_msgs::Pose makePose(const Eigen::Vector3d & rpy, const Eigen::Vector3d & xyz);

geometry_msgs::Pose makePose(const Vector6d & posevec);

Eigen::Affine3d getTransformation(const Vector6d & posevec);

geometry_msgs::PoseStamped makePoseStamped(
  const geometry_msgs::Pose & pose,
  const std::string & frame_id,
  const double time);

geometry_msgs::Pose affineToPose(const Eigen::Affine3d & affine);

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

double timeInSec(const std_msgs::Header & header)
{
  return header.stamp.toSec();
}

template < typename T >
void dropBefore(const double time_second, std::deque < T > & buffer)
{
  while (!buffer.empty()) {
    if (timeInSec(buffer.front().header) >= time_second) {
      break;
    }
    buffer.pop_front();
  }
}

tf::Transform identityTransform()
{
  tf::Transform identity;
  identity.setIdentity();
  return identity;
}

Eigen::Vector3d getXYZ(const PointType & point)
{
  return Eigen::Vector3d(point.x, point.y, point.z);
}

Eigen::MatrixXd rad2deg(const Eigen::MatrixXd & x)
{
  return x * (180.0 / M_PI);
}

pcl::PointCloud < PointType > downsample(
  const pcl::PointCloud < PointType > ::Ptr & input_cloud, const int leaf_size);

template < typename T >
pcl::KdTreeFLANN < T > makeKDTree(const typename pcl::PointCloud < T > ::Ptr & pointcloud) {
  pcl::KdTreeFLANN < T > kdtree;
  kdtree.setInputCloud(pointcloud);
  return kdtree;
}

tf::Pose poseMsgToTF(const geometry_msgs::Pose & msg);

class IMUConverter {
public:
  IMUConverter();
  sensor_msgs::Imu imuConverter(const sensor_msgs::Imu & imu_in) const;

private:
  ros::NodeHandle nh;
  Eigen::Matrix3d extRot;
  Eigen::Quaterniond extQRPY;
};

#endif
