#ifndef CLOUD_OPTIMIZER_HPP_
#define CLOUD_OPTIMIZER_HPP_

#include <tuple>
#include <pcl/point_cloud.h>
#include "point_type.hpp"
#include "matrix_type.h"
#include "kdtree.hpp"

class CloudOptimizer
{
public:
  CloudOptimizer(
    const int N_SCAN,
    const int Horizon_SCAN,
    const int numberOfCores,
    const pcl::PointCloud<PointType> & corner_downsampled,
    const pcl::PointCloud<PointType> & surface_downsampled,
    const pcl::PointCloud<PointType>::Ptr & corner_map_downsampled,
    const pcl::PointCloud<PointType>::Ptr & surface_map_downsampled)
  : N_SCAN(N_SCAN),
    Horizon_SCAN(Horizon_SCAN),
    numberOfCores(numberOfCores),
    corner_downsampled(corner_downsampled),
    surface_downsampled(surface_downsampled),
    corner_map_downsampled(corner_map_downsampled),
    surface_map_downsampled(surface_map_downsampled),
    kdtreeCornerFromMap(KDTree<PointType>(corner_map_downsampled)),
    kdtreeSurfFromMap(KDTree<PointType>(surface_map_downsampled))
  {
  }

  std::tuple<pcl::PointCloud<PointType>, pcl::PointCloud<PointType>>
  run(const Vector6d & posevec) const;

private:
  const int N_SCAN;
  const int Horizon_SCAN;
  const int numberOfCores;
  const pcl::PointCloud<PointType> corner_downsampled;
  const pcl::PointCloud<PointType> surface_downsampled;
  const pcl::PointCloud<PointType>::Ptr corner_map_downsampled;
  const pcl::PointCloud<PointType>::Ptr surface_map_downsampled;
  const KDTree<PointType> kdtreeCornerFromMap;
  const KDTree<PointType> kdtreeSurfFromMap;
};

#endif
