#include "cloud_optimizer.hpp"
#include "homogeneous.h"
#include "utility.hpp"

#include <range/v3/all.hpp>

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

const int n_neighbors = 5;

Eigen::MatrixXd get(
  const pcl::PointCloud<pcl::PointXYZ>::Ptr & pointcloud,
  const std::vector<int> & indices)
{
  Eigen::MatrixXd A(3, indices.size());
  for (const auto & [j, index] : ranges::views::enumerate(indices)) {
    A.col(j) = getXYZ(pointcloud->at(index));
  }
  return A;
}

Eigen::MatrixXd calcCovariance(const Eigen::MatrixXd & X)
{
  const Eigen::Vector3d c = X.rowwise().mean();
  const Eigen::MatrixXd D = X.colwise() - c;
  return D * D.transpose() / X.cols();
}

std::tuple<std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector3d>, std::vector<double>>
CloudOptimizer::run(const Vector6d & posevec) const
{
  const Eigen::Affine3d point_to_map = getTransformation(posevec);
  std::vector<Eigen::Vector3d> edge_coeffs(edge_->size());
  std::vector<double> edge_coeffs_b(edge_->size());
  std::vector<bool> edge_flags(edge_->size(), false);

  // edge optimization
  #pragma omp parallel for num_threads(numberOfCores)
  for (unsigned int i = 0; i < edge_->size(); i++) {
    const pcl::PointXYZ p = transform(point_to_map, edge_->at(i));
    const auto [indices, squared_distances] = edge_kdtree_.nearestKSearch(p, n_neighbors);

    if (squared_distances.back() >= 1.0) {
      continue;
    }

    const Eigen::MatrixXd neighbors = get(edge_map_, indices);
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(calcCovariance(neighbors));
    const Eigen::Vector3d d1 = solver.eigenvalues();
    const Eigen::Matrix3d v1 = solver.eigenvectors();

    if (d1(0) <= 3 * d1(1)) {
      continue;
    }

    const Eigen::Vector3d c = neighbors.rowwise().mean();
    const Eigen::Vector3d p0 = getXYZ(p);
    const Eigen::Vector3d p1 = c + 0.1 * v1.col(0);
    const Eigen::Vector3d p2 = c - 0.1 * v1.col(0);

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
    edge_coeffs[i] = (s / (l12 * a012)) * v;
    edge_coeffs_b[i] = -(s / l12) * a012;
    edge_flags[i] = true;
  }

  std::vector<Eigen::Vector3d> surface_coeffs(surface_->size());
  std::vector<double> surface_coeffs_b(surface_->size());
  std::vector<bool> surface_flags(surface_->size(), false);

  // surface optimization
  #pragma omp parallel for num_threads(numberOfCores)
  for (unsigned int i = 0; i < surface_->size(); i++) {
    const pcl::PointXYZ p = transform(point_to_map, surface_->at(i));
    const auto [indices, squared_distances] = surface_kdtree_.nearestKSearch(p, 5);

    if (squared_distances[4] >= 1.0) {
      continue;
    }

    const Eigen::Matrix<double, 5, 1> b = -1.0 * Eigen::Matrix<double, 5, 1>::Ones();
    const Eigen::Matrix<double, 5, 3> A = makeMatrixA(surface_map_, indices);
    const Eigen::Vector3d x = A.colPivHouseholderQr().solve(b);

    if (!validatePlane(A, x)) {
      continue;
    }

    const Eigen::Vector4d y = toHomogeneous(x);
    const Eigen::Vector4d q = toHomogeneous(getXYZ(p));
    const double pd2 = y.dot(q);
    const double k = fabs(pd2 / x.norm()) / sqrt(getXYZ(p).norm());

    if (k >= 1.0) {
      continue;
    }

    const double s = 1 - 0.9 * k;

    surface_coeffs[i] = (s / x.norm()) * x;
    surface_coeffs_b[i] = -(s / x.norm()) * pd2;
    surface_flags[i] = true;
  }

  std::vector<Eigen::Vector3d> points;
  std::vector<Eigen::Vector3d> coeffs;
  std::vector<double> b;

  for (unsigned int i = 0; i < edge_->size(); ++i) {
    if (edge_flags[i]) {
      points.push_back(getXYZ(edge_->at(i)));
      coeffs.push_back(edge_coeffs[i]);
      b.push_back(edge_coeffs_b[i]);
    }
  }
  for (unsigned int i = 0; i < surface_->size(); ++i) {
    if (surface_flags[i]) {
      points.push_back(getXYZ(surface_->at(i)));
      coeffs.push_back(surface_coeffs[i]);
      b.push_back(surface_coeffs_b[i]);
    }
  }

  return {points, coeffs, b};
}
