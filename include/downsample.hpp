#ifndef DOWNSAMPLE_HPP_
#define DOWNSAMPLE_HPP_

#include <pcl/point_cloud.h>
#include "point_type.hpp"

pcl::PointCloud<PointType>::Ptr downsample(
  const pcl::PointCloud<PointType>::Ptr & input_cloud, const float leaf_size);

#endif
