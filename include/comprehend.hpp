#ifndef COMPREHEND_HPP_
#define COMPREHEND_HPP_

#include <pcl/point_cloud.h>
#include <vector>

template<typename T>
typename pcl::PointCloud<T>::Ptr comprehend(
  const pcl::PointCloud<T> & points,
  const std::vector<int> & indices)
{
  typename pcl::PointCloud<T>::Ptr results(new pcl::PointCloud<T>());
  for (unsigned int index : indices) {
    results->push_back(points.at(index));
  }
  return results;
}

#endif
