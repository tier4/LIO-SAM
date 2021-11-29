#ifndef _JACOBIAN_LIDAR_ODOMETRY_H_
#define _JACOBIAN_LIDAR_ODOMETRY_H_

#include <Eigen/Core>

Eigen::Matrix3d dRdx(const double x, const double y, const double z)
{
  const double sx = sin(x);
  const double cx = cos(x);
  const double sy = sin(y);
  const double cy = cos(y);
  const double sz = sin(z);
  const double cz = cos(z);
  return (Eigen::Matrix3d() <<
         0., sz * sx + sy * cz * cx, +sz * cx - sy * cz * sx,
         0., sy * sz * cx - cz * sx, -cz * cx - sy * sz * sx,
         0., cy * cx, -cy * sx
  ).finished();
}

Eigen::Matrix3d dRdy(const double x, const double y, const double z)
{
  const double sx = sin(x);
  const double cx = cos(x);
  const double sy = sin(y);
  const double cy = cos(y);
  const double sz = sin(z);
  const double cz = cos(z);
  return (Eigen::Matrix3d() <<
         -cy * sz, -cz * cx - sy * sz * sx, cz * sx - sy * sz * cx,
         +cy * cz, +sy * cz * sx - sz * cx, sz * sx + sy * cz * cx,
         0., 0., 0.
  ).finished();
}

Eigen::Matrix3d dRdz(const double x, const double y, const double z)
{
  const double sx = sin(x);
  const double cx = cos(x);
  const double sy = sin(y);
  const double cy = cos(y);
  const double sz = sin(z);
  const double cz = cos(z);
  return (Eigen::Matrix3d() <<
         -sy * cz, cy * cz * sx, +cy * cz * cx,
         -sy * sz, cy * sz * sx, +cy * sz * cx,
         -cy, -sy * sx, -sy * cx
  ).finished();
}

#endif
