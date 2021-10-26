#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>

template<typename T>
pcl::PointCloud<T> getPointCloud(const sensor_msgs::PointCloud2 & roscloud)
{
  pcl::PointCloud<T> pclcloud;
  pcl::fromROSMsg(roscloud, pclcloud);
  return pclcloud;
}
