#ifndef POSE_OPTIMIZER_HPP_
#define POSE_OPTIMIZER_HPP_

#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include "point_type.hpp"
#include "matrix_type.h"
#include "kdtree.hpp"

class PoseOptimizer
{
public:
  PoseOptimizer(const int N_SCAN, const int Horizon_SCAN, const int numberOfCores);

  bool LMOptimization(
    const pcl::PointCloud<PointType> & laserCloudOri,
    const pcl::PointCloud<PointType> & coeffSel,
    const int iterCount, bool & isDegenerate, Vector6d & posevec) const;

  std::tuple<pcl::PointCloud<PointType>, pcl::PointCloud<PointType>>
  optimization(
    const pcl::PointCloud<PointType> & laserCloudCornerLastDS,
    const pcl::PointCloud<PointType> & laserCloudSurfLastDS,
    const KDTree<PointType> & kdtreeCornerFromMap,
    const KDTree<PointType> & kdtreeSurfFromMap,
    const pcl::PointCloud<PointType>::Ptr & laserCloudCornerFromMapDS,
    const pcl::PointCloud<PointType>::Ptr & laserCloudSurfFromMapDS,
    const Vector6d & posevec) const;

  std::tuple<Vector6d, bool> run(
    const pcl::PointCloud<PointType> & laserCloudCornerLastDS,
    const pcl::PointCloud<PointType> & laserCloudSurfLastDS,
    const pcl::PointCloud<PointType>::Ptr & laserCloudCornerFromMapDS,
    const pcl::PointCloud<PointType>::Ptr & laserCloudSurfFromMapDS,
    const Vector6d & initial_posevec) const;

private:
  const int N_SCAN;
  const int Horizon_SCAN;
  const int numberOfCores;
};

#endif
