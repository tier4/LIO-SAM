#ifndef _JACOBIAN_LIDAR_ODOMETRY_H_
#define _JACOBIAN_LIDAR_ODOMETRY_H_

#include <Eigen/Core>

Eigen::Matrix3d dRdx(const Eigen::Vector3d & rpy)
{
  const double sx = sin(rpy(0));
  const double cx = cos(rpy(0));
  const double sy = sin(rpy(1));
  const double cy = cos(rpy(1));
  const double sz = sin(rpy(2));
  const double cz = cos(rpy(2));
  return (Eigen::Matrix3d() <<
         0., sz * sx + sy * cz * cx, +sz * cx - sy * cz * sx,
         0., sy * sz * cx - cz * sx, -cz * cx - sy * sz * sx,
         0., cy * cx, -cy * sx
  ).finished();
}

Eigen::Matrix3d dRdy(const Eigen::Vector3d & rpy)
{
  const double sx = sin(rpy(0));
  const double cx = cos(rpy(0));
  const double sy = sin(rpy(1));
  const double cy = cos(rpy(1));
  const double sz = sin(rpy(2));
  const double cz = cos(rpy(2));
  return (Eigen::Matrix3d() <<
         -sy * cz, cy * cz * sx, +cy * cz * cx,
         -sy * sz, cy * sz * sx, +cy * sz * cx,
         -cy, -sy * sx, -sy * cx
  ).finished();
}

Eigen::Matrix3d dRdz(const Eigen::Vector3d & rpy)
{
  const double sx = sin(rpy(0));
  const double cx = cos(rpy(0));
  const double sy = sin(rpy(1));
  const double cy = cos(rpy(1));
  const double sz = sin(rpy(2));
  const double cz = cos(rpy(2));
  return (Eigen::Matrix3d() <<
         -cy * sz, -cz * cx - sy * sz * sx, cz * sx - sy * sz * cx,
         +cy * cz, +sy * cz * sx - sz * cx, sz * sx + sy * cz * cx,
         0., 0., 0.
  ).finished();
}

#endif
