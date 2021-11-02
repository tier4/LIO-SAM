#include "cloud_optimizer.hpp"
#include "homogeneous.h"
#include "utility.hpp"

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
  const pcl::PointCloud<pcl::PointXYZ>::Ptr & pointcloud,
  const std::vector<int> & indices)
{
  Eigen::Matrix<double, 5, 3> A = Eigen::Matrix<double, 5, 3>::Zero();
  for (int j = 0; j < 5; j++) {
    A.row(j) = getXYZ(pointcloud->at(indices[j]));
  }
  return A;
}

std::tuple<pcl::PointCloud<pcl::PointXYZ>, pcl::PointCloud<pcl::PointXYZI>>
CloudOptimizer::run(const Vector6d & posevec) const
{
  const Eigen::Affine3d point_to_map = getTransformation(posevec);
  std::vector<pcl::PointXYZ> edge_points(N_SCAN * Horizon_SCAN);
  std::vector<pcl::PointXYZI> edge_coeffs(N_SCAN * Horizon_SCAN);
  // edge point holder for parallel computation
  std::vector<bool> edge_flags(N_SCAN * Horizon_SCAN, false);

  // edge optimization
  #pragma omp parallel for num_threads(numberOfCores)
  for (unsigned int i = 0; i < edge_downsampled->size(); i++) {
    const pcl::PointXYZ point = edge_downsampled->at(i);
    const pcl::PointXYZ map_point = transform(point_to_map, point);
    const auto [indices, squared_distances] = edge_kdtree_.nearestKSearch(map_point, 5);

    if (squared_distances[4] >= 1.0) {
      continue;
    }

    Eigen::Vector3d c = Eigen::Vector3d::Zero();
    for (int j = 0; j < 5; j++) {
      c += getXYZ(edge_map_downsampled->at(indices[j]));
    }
    c /= 5.0;

    Eigen::Matrix3d sa = Eigen::Matrix3d::Zero();

    for (int j = 0; j < 5; j++) {
      const Eigen::Vector3d x = getXYZ(edge_map_downsampled->at(indices[j]));
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
    const Eigen::Vector3d p0 = getXYZ(map_point);
    const Eigen::Vector3d p1 = c + 0.1 * v1.row(0).transpose();
    const Eigen::Vector3d p2 = c - 0.1 * v1.row(0).transpose();

    const Eigen::Vector3d d01 = p0 - p1;
    const Eigen::Vector3d d02 = p0 - p2;
    const Eigen::Vector3d d12 = p1 - p2;

    // Cross product version.
    // The current version can be replaced with a simpler computation
    // const Eigen::Vector3d cross(
    //  d01(0) * d02(1) - d02(0) * d01(1),
    //  d01(0) * d02(2) - d02(0) * d01(2),
    //  d01(1) * d02(2) - d02(1) * d01(2));
    const Eigen::Vector3d cross(
      d01(1) * d02(2) - d01(2) * d02(1),
      d01(2) * d02(0) - d01(0) * d02(2),
      d01(0) * d02(1) - d01(1) * d02(0));

    const double a012 = cross.norm();

    const double l12 = d12.norm();

    const Eigen::Vector3d v(
      (d12(1) * cross(2) - cross(2) * d12(1)),
      (d12(2) * cross(0) - cross(0) * d12(2)),
      (d12(0) * cross(1) - cross(1) * d12(0)));

    // This is the auther's version. But this is possibly a bug
    // const Eigen::Vector3d v(
    //   (d12(1) * cross(2) - d12(2) * cross(1)),
    //   (d12(2) * cross(0) - d12(0) * cross(2)),
    //   (d12(0) * cross(1) - d12(1) * cross(0)));

    const double ld2 = a012 / l12;

    const double s = 1 - 0.9 * fabs(ld2);

    if (s <= 0.1) {
      continue;
    }
    edge_points[i] = point;
    edge_coeffs[i] = makePoint(s * v / (a012 * l12), s * ld2);
    edge_flags[i] = true;
  }

  std::vector<pcl::PointXYZ> surface_points(N_SCAN * Horizon_SCAN);
  std::vector<pcl::PointXYZI> surface_coeffs(N_SCAN * Horizon_SCAN);

  // surf point holder for parallel computation
  std::vector<bool> surface_flags(N_SCAN * Horizon_SCAN, false);

  // surface optimization
  #pragma omp parallel for num_threads(numberOfCores)
  for (unsigned int i = 0; i < surface_downsampled->size(); i++) {
    const pcl::PointXYZ point = surface_downsampled->at(i);
    const pcl::PointXYZ map_point = transform(point_to_map, point);
    const auto [indices, squared_distances] = surface_kdtree_.nearestKSearch(map_point, 5);

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
    const Eigen::Vector4d q = toHomogeneous(getXYZ(map_point));
    const float pd2 = y.transpose() * q;
    const float s = 1 - 0.9 * fabs(pd2) / sqrt(getXYZ(map_point).norm());

    if (s <= 0.1) {
      continue;
    }

    surface_points[i] = point;
    surface_coeffs[i] = makePoint((s / x.norm()) * x, s * pd2);
    surface_flags[i] = true;
  }

  pcl::PointCloud<pcl::PointXYZ> points;
  pcl::PointCloud<pcl::PointXYZI> coeffs;

  // combine edge coeffs
  for (unsigned int i = 0; i < edge_downsampled->size(); ++i) {
    if (edge_flags[i]) {
      points.push_back(edge_points[i]);
      coeffs.push_back(edge_coeffs[i]);
    }
  }
  // combine surf coeffs
  for (unsigned int i = 0; i < surface_downsampled->size(); ++i) {
    if (surface_flags[i]) {
      points.push_back(surface_points[i]);
      coeffs.push_back(surface_coeffs[i]);
    }
  }

  return {points, coeffs};
}
