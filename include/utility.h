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

sensor_msgs::PointCloud2 toRosMsg(const pcl::PointCloud < PointType > & pointcloud)
{
  sensor_msgs::PointCloud2 msg;
  pcl::toROSMsg(pointcloud, msg);
  return msg;
}

sensor_msgs::PointCloud2 publishCloud(
  const ros::Publisher & thisPub,
  const pcl::PointCloud < PointType > & thisCloud,
  const ros::Time thisStamp,
  const std::string thisFrame)
{
  sensor_msgs::PointCloud2 tempCloud;
  pcl::toROSMsg(thisCloud, tempCloud);
  tempCloud.header.stamp = thisStamp;
  tempCloud.header.frame_id = thisFrame;
  if (thisPub.getNumSubscribers() != 0) {
    thisPub.publish(tempCloud);
  }
  return tempCloud;
}

Eigen::Vector3d pointToEigen(const geometry_msgs::Point & p)
{
  return Eigen::Vector3d(p.x, p.y, p.z);
}

Eigen::Vector3d vector3ToEigen(const geometry_msgs::Vector3 & p)
{
  return Eigen::Vector3d(p.x, p.y, p.z);
}

PointType makePoint(const Eigen::Vector3d & point, const float intensity = 0.0)
{
  const Eigen::Vector3f q = point.cast < float > ();
  PointType p;
  p.x = q(0);
  p.y = q(1);
  p.z = q(2);
  p.intensity = intensity;
  return p;
}

Eigen::Affine3d makeAffine(
  const Eigen::Vector3d & rpy = Eigen::Vector3d::Zero(),
  const Eigen::Vector3d & point = Eigen::Vector3d::Zero())
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

Eigen::Quaterniond quaternionToEigen(const geometry_msgs::Quaternion & quat_msg)
{
  Eigen::Quaterniond quat_eigen;
  tf::quaternionMsgToEigen(quat_msg, quat_eigen);
  return quat_eigen;
}

geometry_msgs::Quaternion eigenToQuaternion(const Eigen::Quaterniond & quat_eigen)
{
  geometry_msgs::Quaternion quat_msg;
  tf::quaternionEigenToMsg(quat_eigen, quat_msg);
  return quat_msg;
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

std::tuple < Eigen::Vector3d, Eigen::Vector3d > getXYZRPY(const Eigen::Affine3d & affine)
{
  double x, y, z, roll, pitch, yaw;
  pcl::getTranslationAndEulerAngles(affine, x, y, z, roll, pitch, yaw);
  return {Eigen::Vector3d(x, y, z), Eigen::Vector3d(roll, pitch, yaw)};
}

geometry_msgs::Point eigenToPoint(const Eigen::Vector3d & v)
{
  geometry_msgs::Point p;
  p.x = v[0];
  p.y = v[1];
  p.z = v[2];
  return p;
}

geometry_msgs::Pose makePose(
  const geometry_msgs::Quaternion & orientation,
  const geometry_msgs::Point & position)
{
  geometry_msgs::Pose pose;
  pose.position = position;
  pose.orientation = orientation;
  return pose;
}

geometry_msgs::Twist makeTwist(
  const geometry_msgs::Vector3 & angular,
  const geometry_msgs::Vector3 & linear)
{
  geometry_msgs::Twist twist;
  twist.angular = angular;
  twist.linear = linear;
  return twist;
}

geometry_msgs::Pose makePose(const Eigen::Vector3d & rpy, const Eigen::Vector3d & xyz)
{
  const auto orientation = tf::createQuaternionMsgFromRollPitchYaw(rpy(0), rpy(1), rpy(2));
  const auto position = eigenToPoint(xyz);
  return makePose(orientation, position);
}

geometry_msgs::Pose makePose(const Vector6d & posevec)
{
  return makePose(posevec.head(3), posevec.tail(3));
}

Eigen::Affine3d getTransformation(const Vector6d & posevec)
{
  Eigen::Affine3d transform;
  pcl::getTransformation(
    posevec(3), posevec(4), posevec(5),
    posevec(0), posevec(1), posevec(2), transform);
  return transform;
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

  sensor_msgs::Imu imuConverter(const sensor_msgs::Imu & imu_in) const
  {
    sensor_msgs::Imu imu_out = imu_in;
    // rotate acceleration
    const Eigen::Vector3d acc = vector3ToEigen(imu_in.linear_acceleration);
    imu_out.linear_acceleration = eigenToVector3(extRot * acc);

    const Eigen::Vector3d gyr = vector3ToEigen(imu_in.angular_velocity);
    imu_out.angular_velocity = eigenToVector3(extRot * gyr);

    const Eigen::Quaterniond q_from = quaternionToEigen(imu_in.orientation);
    const Eigen::Quaterniond q_final = q_from * extQRPY;
    imu_out.orientation = eigenToQuaternion(q_final);

    if (q_final.norm() < 0.1) {
      throw std::runtime_error("Invalid quaternion, please use a 9-axis IMU!");
    }

    return imu_out;
  }

private:
  ros::NodeHandle nh;
  Eigen::Matrix3d extRot;
  Eigen::Quaterniond extQRPY;
};

#endif
