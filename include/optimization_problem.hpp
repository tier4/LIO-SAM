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
    const pcl::PointCloud<pcl::PointXYZ>::Ptr & edge_,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr & surface_,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr & edge_map_,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr & surface_map_)
  : numberOfCores(numberOfCores),
    edge_(edge_),
    surface_(surface_),
    edge_map_(edge_map_),
    surface_map_(surface_map_),
    edge_kdtree_(KDTree<pcl::PointXYZ>(edge_map_)),
    surface_kdtree_(KDTree<pcl::PointXYZ>(surface_map_))
  {
  }

  std::tuple<std::vector<Eigen::Vector3d>, std::vector<double>, std::vector<bool>>
  fromEdge(const Eigen::Affine3d & point_to_map) const;

  std::tuple<std::vector<Eigen::Vector3d>, std::vector<double>, std::vector<bool>>
  fromSurface(const Eigen::Affine3d & point_to_map) const;

  std::tuple<Eigen::MatrixXd, Eigen::VectorXd> run(const Vector6d & posevec) const;

private:
  const int numberOfCores;
  const pcl::PointCloud<pcl::PointXYZ>::Ptr edge_;
  const pcl::PointCloud<pcl::PointXYZ>::Ptr surface_;
  const pcl::PointCloud<pcl::PointXYZ>::Ptr edge_map_;
  const pcl::PointCloud<pcl::PointXYZ>::Ptr surface_map_;
  const KDTree<pcl::PointXYZ> edge_kdtree_;
  const KDTree<pcl::PointXYZ> surface_kdtree_;
};

bool isDegenerate(const OptimizationProblem & problem, const Vector6d & posevec);

Vector6d optimizePose(const OptimizationProblem & problem, const Vector6d & initial_posevec);

#endif
