#include "cloud_optimizer.hpp"
#include "homogeneous.h"
#include "utility.h"

bool validatePlane(
  const Eigen::Matrix<double, 5, 3> & A,
  const Eigen::Vector3d & x)
{
  const Eigen::Vector4d y = toHomogeneous(x) / x.norm();

  for (int j = 0; j < 5; j++) {
    const Eigen::Vector3d p = A.row(j);
    const Eigen::Vector4d q = toHomogeneous(p);

    if (fabs(y.transpose() * q) > 0.2) {
      return false;
    }
  }
  return true;
}

Eigen::Matrix<double, 5, 3> makeMatrixA(
  const pcl::PointCloud<PointType>::Ptr & pointcloud,
  const std::vector<int> & indices)
{
  Eigen::Matrix<double, 5, 3> A = Eigen::Matrix<double, 5, 3>::Zero();
  for (int j = 0; j < 5; j++) {
    A.row(j) = getXYZ(pointcloud->at(indices[j]));
  }
  return A;
}

std::tuple<pcl::PointCloud<PointType>, pcl::PointCloud<PointType>>
CloudOptimizer::run(const Vector6d & posevec) const
{
  const Eigen::Affine3d point_to_map = getTransformation(posevec);
  std::vector<PointType> laserCloudOriCornerVec(N_SCAN * Horizon_SCAN);
  std::vector<PointType> coeffSelCornerVec(N_SCAN * Horizon_SCAN);
  // corner point holder for parallel computation
  std::vector<bool> laserCloudOriCornerFlag(N_SCAN * Horizon_SCAN, false);

  // corner optimization
  #pragma omp parallel for num_threads(numberOfCores)
  for (unsigned int i = 0; i < corner_downsampled.size(); i++) {
    const PointType point = corner_downsampled.at(i);
    const Eigen::Vector3d map_point = point_to_map * getXYZ(point);
    const PointType p = makePoint(map_point, point.intensity);
    const auto [indices, squared_distances] = kdtreeCornerFromMap.nearestKSearch(p, 5);

    if (squared_distances[4] >= 1.0) {
      continue;
    }

    Eigen::Vector3d c = Eigen::Vector3d::Zero();
    for (int j = 0; j < 5; j++) {
      c += getXYZ(corner_map_downsampled->at(indices[j]));
    }
    c /= 5.0;

    Eigen::Matrix3d sa = Eigen::Matrix3d::Zero();

    for (int j = 0; j < 5; j++) {
      const Eigen::Vector3d x = getXYZ(corner_map_downsampled->at(indices[j]));
      const Eigen::Vector3d a = x - c;
      sa += a * a.transpose();
    }

    sa = sa / 5.0;

    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(sa);
    const Eigen::Vector3d d1 = solver.eigenvalues();
    const Eigen::Matrix3d v1 = solver.eigenvectors();

    if (d1(0) <= 3 * d1(1)) {
      continue;
    }
    const Eigen::Vector3d p0 = map_point;
    const Eigen::Vector3d p1 = c + 0.1 * v1.row(0).transpose();
    const Eigen::Vector3d p2 = c - 0.1 * v1.row(0).transpose();

    const Eigen::Vector3d d01 = p0 - p1;
    const Eigen::Vector3d d02 = p0 - p2;
    const Eigen::Vector3d d12 = p1 - p2;

    // const Eigen::Vector3d d012(d01(0) * d02(1) - d02(0) * d01(1),
    //                            d01(0) * d02(2) - d02(0) * d01(2),
    //                            d01(1) * d02(2) - d02(1) * d01(2));
    const Eigen::Vector3d cross(
      d01(1) * d02(2) - d01(2) * d02(1),
      d01(2) * d02(0) - d01(0) * d02(2),
      d01(0) * d02(1) - d01(1) * d02(0));

    const double a012 = cross.norm();

    const double l12 = d12.norm();

    // possible bag. maybe the commented one is correct
    // const Eigen::Vector3d v(
    //   (d12(1) * cross(2) - cross(2) * d12(1)),
    //   (d12(2) * cross(0) - cross(0) * d12(2)),
    //   (d12(0) * cross(1) - cross(1) * d12(0)));

    const Eigen::Vector3d v(
      (d12(1) * cross(2) - d12(2) * cross(1)),
      (d12(2) * cross(0) - d12(0) * cross(2)),
      (d12(0) * cross(1) - d12(1) * cross(0)));

    const double ld2 = a012 / l12;

    const double s = 1 - 0.9 * fabs(ld2);

    if (s <= 0.1) {
      continue;
    }
    laserCloudOriCornerVec[i] = point;
    coeffSelCornerVec[i] = makePoint(s * v / (a012 * l12), s * ld2);
    laserCloudOriCornerFlag[i] = true;
  }

  std::vector<PointType> laserCloudOriSurfVec(N_SCAN * Horizon_SCAN);
  std::vector<PointType> coeffSelSurfVec(N_SCAN * Horizon_SCAN);

  // surf point holder for parallel computation
  std::vector<bool> laserCloudOriSurfFlag(N_SCAN * Horizon_SCAN, false);

  // surface optimization
  #pragma omp parallel for num_threads(numberOfCores)
  for (unsigned int i = 0; i < surface_downsampled.size(); i++) {
    const PointType point = surface_downsampled.at(i);
    const Eigen::Vector3d map_point = point_to_map * getXYZ(point);
    const PointType p = makePoint(map_point, point.intensity);
    const auto [indices, squared_distances] = kdtreeSurfFromMap.nearestKSearch(p, 5);

    if (squared_distances[4] >= 1.0) {
      continue;
    }

    const Eigen::Matrix<double, 5, 1> b = -1.0 * Eigen::Matrix<double, 5, 1>::Ones();
    const Eigen::Matrix<double, 5, 3> A = makeMatrixA(surface_map_downsampled, indices);
    const Eigen::Vector3d x = A.colPivHouseholderQr().solve(b);

    if (!validatePlane(A, x)) {
      continue;
    }

    const Eigen::Vector4d y = toHomogeneous(x) / x.norm();
    const Eigen::Vector4d q = toHomogeneous(map_point);
    const float pd2 = y.transpose() * q;
    const float s = 1 - 0.9 * fabs(pd2) / sqrt(map_point.norm());

    if (s <= 0.1) {
      continue;
    }

    laserCloudOriSurfVec[i] = point;
    coeffSelSurfVec[i] = makePoint((s / x.norm()) * x, s * pd2);
    laserCloudOriSurfFlag[i] = true;
  }

  pcl::PointCloud<PointType> laserCloudOri;
  pcl::PointCloud<PointType> coeffSel;

  // combine corner coeffs
  for (unsigned int i = 0; i < corner_downsampled.size(); ++i) {
    if (laserCloudOriCornerFlag[i]) {
      laserCloudOri.push_back(laserCloudOriCornerVec[i]);
      coeffSel.push_back(coeffSelCornerVec[i]);
    }
  }
  // combine surf coeffs
  for (unsigned int i = 0; i < surface_downsampled.size(); ++i) {
    if (laserCloudOriSurfFlag[i]) {
      laserCloudOri.push_back(laserCloudOriSurfVec[i]);
      coeffSel.push_back(coeffSelSurfVec[i]);
    }
  }

  return {laserCloudOri, coeffSel};
}
