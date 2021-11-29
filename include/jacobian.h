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
         0., sy * sx + sz * cy * cx, +sy * cx - sz * cy * sx,
         0., sz * sy * cx - cy * sx, -cy * cx - sz * sy * sx,
         0., cz * cx, -cz * sx
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
         -cz * sy, -cy * cx - sz * sy * sx, cy * sx - sz * sy * cx,
         +cz * cy, +sz * cy * sx - sy * cx, sy * sx + sz * cy * cx,
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
         -sz * cy, cz * cy * sx, +cz * cy * cx,
         -sz * sy, cz * sy * sx, +cz * sy * cx,
         -cz, -sz * sx, -sz * cx
  ).finished();
}

#endif
