#include "utility.h"

pcl::PointCloud<PointType> downsample(
  const pcl::PointCloud<PointType>::Ptr & input_cloud, const int leaf_size)
{
  pcl::VoxelGrid<PointType> filter;
  pcl::PointCloud<PointType> downsampled;

  filter.setLeafSize(leaf_size, leaf_size, leaf_size);
  filter.setInputCloud(input_cloud);
  filter.filter(downsampled);

  return downsampled;
}
