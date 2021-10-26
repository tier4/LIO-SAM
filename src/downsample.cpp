#include "downsample.hpp"

#include <pcl/filters/voxel_grid.h>

pcl::PointCloud<PointType>::Ptr downsample(
  const pcl::PointCloud<PointType>::Ptr & input_cloud, const float leaf_size)
{
  pcl::VoxelGrid<PointType> filter;
  pcl::PointCloud<PointType>::Ptr downsampled(new pcl::PointCloud<PointType>());

  filter.setLeafSize(leaf_size, leaf_size, leaf_size);
  filter.setInputCloud(input_cloud);
  filter.filter(*downsampled);

  return downsampled;
}
