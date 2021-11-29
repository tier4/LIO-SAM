#ifndef _JACOBIAN_LIDAR_ODOMETRY_H_
#define _JACOBIAN_LIDAR_ODOMETRY_H_

#include <Eigen/Core>

Eigen::Matrix3d dRdz(const double x, const double y, const double z)
{
  const double sx = sin(x);
  const double cx = cos(x);
  const double sy = sin(y);
  const double cy = cos(y);
  const double sz = sin(z);
  const double cz = cos(z);
  return (Eigen::Matrix3d() <<
         cz * sy * sx, +cz * sy * cx, -sz * sy,
         -sz * sx, -sz * cx, -cz,
         cz * cy * sx, +cz * cy * cx, -sz * cy
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
         +sz * cy * sx - sy * cx, sy * sx + sz * cy * cx, +cz * cy,
         0., 0., 0.,
         -cy * cx - sz * sy * sx, cy * sx - sz * sy * cx, -cz * sy
  ).finished();
}

Eigen::Matrix3d dRdx(const double x, const double y, const double z)
{
  const double sx = sin(x);
  const double cx = cos(x);
  const double sy = sin(y);
  const double cy = cos(y);
  const double sz = sin(z);
  const double cz = cos(z);
  return (Eigen::Matrix3d() <<
         sz * sy * cx - cy * sx, -cy * cx - sz * sy * sx, 0.,
         cz * cx, -cz * sx, 0.,
         sy * sx + sz * cy * cx, +sy * cx - sz * cy * sx, 0.
  ).finished();
}

#endif
