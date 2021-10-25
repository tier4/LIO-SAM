#ifndef POSE_OPTIMIZER_HPP_
#define POSE_OPTIMIZER_HPP_

#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include "cloud_optimizer.hpp"
#include "point_type.hpp"
#include "matrix_type.h"
#include "kdtree.hpp"

std::tuple<Vector6d, bool> optimizePose(
  const CloudOptimizer & cloud_optimizer,
  const Vector6d & initial_posevec);

#endif
