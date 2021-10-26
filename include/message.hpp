#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>

template<typename T>
typename pcl::PointCloud<T>::Ptr getPointCloud(const sensor_msgs::PointCloud2 & roscloud)
{
  typename pcl::PointCloud<T>::Ptr pclcloud(new pcl::PointCloud<T>());
  pcl::fromROSMsg(roscloud, *pclcloud);
  return pclcloud;
}
