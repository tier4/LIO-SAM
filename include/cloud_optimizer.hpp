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
    const int numberOfCores,
    const int edgeFeatureMinValidNum,
    const int surfFeatureMinValidNum,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr & edge_downsampled,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr & surface_downsampled,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr & edge_map_downsampled,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr & surface_map_downsampled)
  : numberOfCores(numberOfCores),
    edge_downsampled(edge_downsampled),
    surface_downsampled(surface_downsampled),
    edge_map_downsampled(edge_map_downsampled),
    surface_map_downsampled(surface_map_downsampled),
    edge_kdtree_(KDTree<pcl::PointXYZ>(edge_map_downsampled)),
    surface_kdtree_(KDTree<pcl::PointXYZ>(surface_map_downsampled))
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

  std::tuple<std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector3d>, std::vector<double>>
  run(const Vector6d & posevec) const;

private:
  const int numberOfCores;
  const pcl::PointCloud<pcl::PointXYZ>::Ptr edge_downsampled;
  const pcl::PointCloud<pcl::PointXYZ>::Ptr surface_downsampled;
  const pcl::PointCloud<pcl::PointXYZ>::Ptr edge_map_downsampled;
  const pcl::PointCloud<pcl::PointXYZ>::Ptr surface_map_downsampled;
  const KDTree<pcl::PointXYZ> edge_kdtree_;
  const KDTree<pcl::PointXYZ> surface_kdtree_;
};

#endif
