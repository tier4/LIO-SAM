#ifndef _JACOBIAN_LIDAR_ODOMETRY_H_
#define _JACOBIAN_LIDAR_ODOMETRY_H_

#include <Eigen/Core>

Eigen::Matrix3f dRdx(const float x, const float y, const float z) {
  const float sx = sin(x);
  const float cx = cos(x);
  const float sy = sin(y);
  const float cy = cos(y);
  const float sz = sin(z);
  const float cz = cos(z);
  return (Eigen::Matrix3f() <<
       cz*sy*sx, + cz*sy*cx, - sz*sy,
      -sz   *sx, - sz   *cx, -    cz,
       cz*cy*sx, + cz*cy*cx, - sz*cy
  ).finished();
}

Eigen::Matrix3f dRdy(const float x, const float y, const float z) {
  const float sx = sin(x);
  const float cx = cos(x);
  const float sy = sin(y);
  const float cy = cos(y);
  const float sz = sin(z);
  const float cz = cos(z);
  return (Eigen::Matrix3f() <<
      +sz*cy*sx - sy*cx, sy*sx + sz*cy*cx, + cz*cy,
                     0.,               0.,      0.,
      -cy*cx - sz*sy*sx, cy*sx - sz*sy*cx, - cz*sy
  ).finished();
}

Eigen::Matrix3f dRdz(const float x, const float y, const float z) {
  const float sx = sin(x);
  const float cx = cos(x);
  const float sy = sin(y);
  const float cy = cos(y);
  const float sz = sin(z);
  const float cz = cos(z);
  return (Eigen::Matrix3f() <<
      sz*sy*cx - cy*sx, -cy*cx-sz*sy*sx, 0.,
      cz*cx           , -cz*sx         , 0.,
      sy*sx + sz*cy*cx, +sy*cx-sz*cy*sx, 0.
  ).finished();
}

#endif
