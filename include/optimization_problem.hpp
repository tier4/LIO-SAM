#ifndef CLOUD_OPTIMIZER_HPP_
#define CLOUD_OPTIMIZER_HPP_

#include <tuple>

#include <fmt/format.h>
#include <pcl/point_cloud.h>

#include "matrix_type.h"
#include "kdtree.hpp"

class OptimizationProblem
{
public:
  OptimizationProblem(
    const int numberOfCores,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr & edge_scan,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr & surface_scan,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr & edge_map,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr & surface_map)
  : numberOfCores(numberOfCores),
    edge_scan_(edge_scan),
    surface_scan_(surface_scan),
    edge_map_(edge_map),
    surface_map_(surface_map),
    edge_kdtree_(KDTree<pcl::PointXYZ>(edge_map)),
    surface_kdtree_(KDTree<pcl::PointXYZ>(surface_map))
  {
  }

  std::tuple<std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector3d>, std::vector<double>>
  fromEdge(const Eigen::Affine3d & point_to_map) const;

  std::tuple<std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector3d>, std::vector<double>>
  fromSurface(const Eigen::Affine3d & point_to_map) const;

  std::tuple<Eigen::MatrixXd, Eigen::VectorXd> run(const Vector6d & posevec) const;

private:
  const int numberOfCores;
  const pcl::PointCloud<pcl::PointXYZ>::Ptr edge_scan_;
  const pcl::PointCloud<pcl::PointXYZ>::Ptr surface_scan_;
  const pcl::PointCloud<pcl::PointXYZ>::Ptr edge_map_;
  const pcl::PointCloud<pcl::PointXYZ>::Ptr surface_map_;
  const KDTree<pcl::PointXYZ> edge_kdtree_;
  const KDTree<pcl::PointXYZ> surface_kdtree_;
};

bool isDegenerate(const OptimizationProblem & problem, const Vector6d & posevec);

Vector6d optimizePose(const OptimizationProblem & problem, const Vector6d & initial_posevec);

#endif
