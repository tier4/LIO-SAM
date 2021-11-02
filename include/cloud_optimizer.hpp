#ifndef CLOUD_OPTIMIZER_HPP_
#define CLOUD_OPTIMIZER_HPP_

#include <tuple>

#include <fmt/format.h>
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
    const int edgeFeatureMinValidNum,
    const int surfFeatureMinValidNum,
    const pcl::PointCloud<PointType>::Ptr & edge_downsampled,
    const pcl::PointCloud<PointType>::Ptr & surface_downsampled,
    const pcl::PointCloud<PointType>::Ptr & edge_map_downsampled,
    const pcl::PointCloud<PointType>::Ptr & surface_map_downsampled)
  : N_SCAN(N_SCAN),
    Horizon_SCAN(Horizon_SCAN),
    numberOfCores(numberOfCores),
    edge_downsampled(edge_downsampled),
    surface_downsampled(surface_downsampled),
    edge_map_downsampled(edge_map_downsampled),
    surface_map_downsampled(surface_map_downsampled),
    kdtreeEdgeFromMap(KDTree<PointType>(edge_map_downsampled)),
    kdtreeSurfFromMap(KDTree<PointType>(surface_map_downsampled))
  {
    if (
      static_cast<int>(edge_downsampled->size()) <= edgeFeatureMinValidNum ||
      static_cast<int>(surface_downsampled->size()) <= surfFeatureMinValidNum)
    {
      throw std::runtime_error(
              fmt::format(
                "Not enough features! Only %d edge and %d planar features available.",
                edge_downsampled->size(), surface_downsampled->size()));
    }
  }

  std::tuple<pcl::PointCloud<PointType>, pcl::PointCloud<PointType>>
  run(const Vector6d & posevec) const;

private:
  const int N_SCAN;
  const int Horizon_SCAN;
  const int numberOfCores;
  const pcl::PointCloud<PointType>::Ptr edge_downsampled;
  const pcl::PointCloud<PointType>::Ptr surface_downsampled;
  const pcl::PointCloud<PointType>::Ptr edge_map_downsampled;
  const pcl::PointCloud<PointType>::Ptr surface_map_downsampled;
  const KDTree<PointType> kdtreeEdgeFromMap;
  const KDTree<PointType> kdtreeSurfFromMap;
};

#endif
