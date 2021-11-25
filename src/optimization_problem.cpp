#include "optimization_problem.hpp"
#include "homogeneous.h"
#include "utility.hpp"
#include "jacobian.h"

#include <Eigen/Eigenvalues>

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

Eigen::VectorXd solveLinear(const Eigen::MatrixXd & A, const Eigen::VectorXd & b)
{
  return A.householderQr().solve(b);
}

bool checkConvergence(const Vector6d & dx)
{
  const float dr = rad2deg(dx.head(3)).norm();
  const float dt = (100 * dx.tail(3)).norm();
  return dr < 0.05 && dt < 0.05;
}

std::tuple<std::vector<Eigen::Vector3d>, std::vector<double>, std::vector<bool>>
OptimizationProblem::fromEdge(const Eigen::Affine3d & point_to_map) const
{
  std::vector<Eigen::Vector3d> coeffs(edge_scan_->size());
  std::vector<double> b(edge_scan_->size());
  std::vector<bool> flags(edge_scan_->size(), false);

  #pragma omp parallel for num_threads(numberOfCores)
  for (unsigned int i = 0; i < edge_scan_->size(); i++) {
    const pcl::PointXYZ p = transform(point_to_map, edge_scan_->at(i));
    const auto [indices, squared_distances] = edge_kdtree_.nearestKSearch(p, n_neighbors);
    if (squared_distances.back() >= 1.0) {
      continue;
    }

    const Eigen::MatrixXd neighbors = get(edge_map_, indices);
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(calcCovariance(neighbors));
    const Eigen::Vector3d eigenvalues = solver.eigenvalues();
    const Eigen::Vector3d eigenvector = solver.eigenvectors().col(2);

    if (eigenvalues(2) <= 3 * eigenvalues(1)) {
      continue;
    }

    const Eigen::Vector3d c = neighbors.rowwise().mean();
    const Eigen::Vector3d p0 = getXYZ(p);
    const Eigen::Vector3d p1 = c + 0.1 * eigenvector;
    const Eigen::Vector3d p2 = c - 0.1 * eigenvector;

    const Eigen::Vector3d d01 = p0 - p1;
    const Eigen::Vector3d d12 = p1 - p2;
    const Eigen::Vector3d d20 = p2 - p0;

    const Eigen::Vector3d u = d20.cross(d01);
    const Eigen::Vector3d v = d12.cross(u);
    const double k = u.norm();

    if (k >= 1.0) {
      continue;
    }

    coeffs[i] = k * v;
    b[i] = -k;
    flags[i] = true;
  }
  return {coeffs, b, flags};
}

std::tuple<std::vector<Eigen::Vector3d>, std::vector<double>, std::vector<bool>>
OptimizationProblem::fromSurface(const Eigen::Affine3d & point_to_map) const
{
  std::vector<Eigen::Vector3d> coeffs(surface_scan_->size());
  std::vector<double> b(surface_scan_->size());
  std::vector<bool> flags(surface_scan_->size(), false);

  // surface optimization
  #pragma omp parallel for num_threads(numberOfCores)
  for (unsigned int i = 0; i < surface_scan_->size(); i++) {
    const pcl::PointXYZ p = transform(point_to_map, surface_scan_->at(i));
    const auto [indices, squared_distances] = surface_kdtree_.nearestKSearch(p, n_neighbors);

    if (squared_distances.back() >= 1.0) {
      continue;
    }

    const Eigen::VectorXd g = -1.0 * Eigen::VectorXd::Ones(n_neighbors);
    const Eigen::MatrixXd A = makeMatrixA(surface_map_, indices);
    const Eigen::Vector3d x = solveLinear(A, g);

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

    coeffs[i] = (s / x.norm()) * x;
    b[i] = -(s / x.norm()) * pd2;
    flags[i] = true;
  }
  return {coeffs, b, flags};
}

Eigen::MatrixXd makeMatrixA(
  const std::vector<Eigen::Vector3d> & points,
  const std::vector<Eigen::Vector3d> & coeffs,
  const Eigen::Vector3d & rpy)
{
  const Eigen::Matrix3d MX = dRdx(rpy(0), rpy(2), rpy(1));
  const Eigen::Matrix3d MY = dRdy(rpy(0), rpy(2), rpy(1));
  const Eigen::Matrix3d MZ = dRdz(rpy(0), rpy(2), rpy(1));

  Eigen::MatrixXd A(points.size(), 6);
  for (unsigned int i = 0; i < points.size(); i++) {
    // in camera

    const Eigen::Vector3d p = points.at(i);
    const Eigen::Vector3d c = coeffs.at(i);
    const Eigen::Vector3d point_ori(p(1), p(2), p(0));
    const Eigen::Vector3d coeff_vec(c(1), c(2), c(0));

    const float arx = coeff_vec.dot(MX * point_ori);
    const float ary = coeff_vec.dot(MY * point_ori);
    const float arz = coeff_vec.dot(MZ * point_ori);

    // lidar -> camera
    A(i, 0) = arz;
    A(i, 1) = arx;
    A(i, 2) = ary;
    A(i, 3) = c(0);
    A(i, 4) = c(1);
    A(i, 5) = c(2);
  }
  return A;
}

std::tuple<Eigen::MatrixXd, Eigen::VectorXd>
OptimizationProblem::run(const Vector6d & posevec) const
{
  const Eigen::Affine3d point_to_map = getTransformation(posevec);
  const auto [edge_coeffs, edge_coeffs_b, edge_flags] = fromEdge(point_to_map);
  const auto [surface_coeffs, surface_coeffs_b, surface_flags] = fromSurface(point_to_map);

  auto edge_indices =
    ranges::views::iota(0, static_cast<int>(edge_scan_->size())) |
    ranges::views::filter([&](int i) {return edge_flags[i];});
  auto surface_indices =
    ranges::views::iota(0, static_cast<int>(surface_scan_->size())) |
    ranges::views::filter([&](int i) {return surface_flags[i];});

  const auto points = ranges::views::concat(
    edge_indices | ranges::views::transform([&](int i) {return getXYZ(edge_scan_->at(i));}),
    surface_indices | ranges::views::transform([&](int i) {return getXYZ(surface_scan_->at(i));})
    ) | ranges::to_vector;

  const auto coeffs = ranges::views::concat(
    edge_indices | ranges::views::transform([&](int i) {return edge_coeffs[i];}),
    surface_indices | ranges::views::transform([&](int i) {return surface_coeffs[i];})
    ) | ranges::to_vector;

  auto b_vector = ranges::views::concat(
    edge_indices | ranges::views::transform([&](int i) {return edge_coeffs_b[i];}),
    surface_indices | ranges::views::transform([&](int i) {return surface_coeffs_b[i];})
    ) | ranges::to_vector;

  const Eigen::MatrixXd A = makeMatrixA(points, coeffs, posevec.head(3));
  const Eigen::Map<Eigen::VectorXd> b(b_vector.data(), b_vector.size());
  return {A, b};
}

bool isDegenerate(const OptimizationProblem & problem, const Vector6d & posevec)
{
  const auto [A, b] = problem.run(posevec);
  const Eigen::MatrixXd AtA = A.transpose() * A;
  const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(AtA);
  const Eigen::VectorXd eigenvalues = es.eigenvalues();
  return (eigenvalues.array() < 100.0).any();
}

Eigen::VectorXd calcUpdate(const Eigen::MatrixXd & A, const Eigen::VectorXd & b)
{
  const Eigen::MatrixXd AtA = A.transpose() * A;
  const Eigen::VectorXd AtB = A.transpose() * b;
  return solveLinear(AtA, AtB);
}

// This optimization is from the original loam_velodyne by Ji Zhang,
// need to cope with coordinate transformation
// lidar <- camera      ---     camera <- lidar
// x = z                ---     x = y
// y = x                ---     y = z
// z = y                ---     z = x
// roll = yaw           ---     roll = pitch
// pitch = roll         ---     pitch = yaw
// yaw = pitch          ---     yaw = roll

Vector6d optimizePose(const OptimizationProblem & problem, const Vector6d & initial_posevec)
{
  Vector6d posevec = initial_posevec;
  for (int iter = 0; iter < 30; iter++) {
    const auto [A, b] = problem.run(posevec);
    if (A.rows() < 50) {
      continue;
    }

    const Eigen::VectorXd dx = calcUpdate(A, b);

    posevec += dx;

    if (checkConvergence(dx)) {
      break;
    }
  }
  return posevec;
}
