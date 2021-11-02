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

std::tuple<pcl::PointCloud<pcl::PointXYZ>, std::vector<Eigen::Vector3d>, std::vector<double>>
CloudOptimizer::run(const Vector6d & posevec) const
{
  const Eigen::Affine3d point_to_map = getTransformation(posevec);
  std::vector<pcl::PointXYZ> edge_points(edge_downsampled->size());
  std::vector<Eigen::Vector3d> edge_coeffs(edge_downsampled->size());
  std::vector<double> edge_coeffs_b(edge_downsampled->size());
  std::vector<bool> edge_flags(edge_downsampled->size(), false);

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

    const double k = fabs(a012 / l12);
    if (k >= 1.0) {
      continue;
    }

    const double s = 1 - 0.9 * k;
    edge_points[i] = point;
    edge_coeffs[i] = (s / (l12 * a012)) * v;
    edge_coeffs_b[i] = -(s / l12) * a012;
    edge_flags[i] = true;
  }

  std::vector<pcl::PointXYZ> surface_points(surface_downsampled->size());
  std::vector<Eigen::Vector3d> surface_coeffs(surface_downsampled->size());
  std::vector<double> surface_coeffs_b(surface_downsampled->size());
  std::vector<bool> surface_flags(surface_downsampled->size(), false);

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

    const Eigen::Vector4d y = toHomogeneous(x);
    const Eigen::Vector4d q = toHomogeneous(getXYZ(map_point));
    const double pd2 = y.dot(q);
    const double k = fabs(pd2 / x.norm()) / sqrt(getXYZ(map_point).norm());

    if (k >= 1.0) {
      continue;
    }

    const double s = 1 - 0.9 * k;

    surface_points[i] = point;
    surface_coeffs[i] = (s / x.norm()) * x;
    surface_coeffs_b[i] = -(s / x.norm()) * pd2;
    surface_flags[i] = true;
  }

  pcl::PointCloud<pcl::PointXYZ> points;
  std::vector<Eigen::Vector3d> coeffs;
  std::vector<double> b;

  for (unsigned int i = 0; i < edge_downsampled->size(); ++i) {
    if (edge_flags[i]) {
      points.push_back(edge_points[i]);
      coeffs.push_back(edge_coeffs[i]);
      b.push_back(edge_coeffs_b[i]);
    }
  }
  for (unsigned int i = 0; i < surface_downsampled->size(); ++i) {
    if (surface_flags[i]) {
      points.push_back(surface_points[i]);
      coeffs.push_back(surface_coeffs[i]);
      b.push_back(surface_coeffs_b[i]);
    }
  }

  return {points, coeffs, b};
}
