#include "optimization_problem.hpp"
#include "homogeneous.h"
#include "utility.hpp"
#include "jacobian.h"

#include <Eigen/Eigenvalues>

#include <range/v3/all.hpp>

bool validatePlane(const Eigen::MatrixXd & X, const Eigen::Vector3d & w)
{
  for (int j = 0; j < X.rows(); j++) {
    const Eigen::Vector3d x = X.row(j);
    if (fabs(w.dot(x) + 1.0) / w.norm() > 0.2) {
      return false;
    }
  }
  return true;
}

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

const int n_neighbors = 5;

std::vector<int> filteredIndices(const std::vector<bool> & flags)
{
  return ranges::views::iota(0, static_cast<int>(flags.size())) |
         ranges::views::filter([&](int i) {return flags[i];}) |
         ranges::to_vector;
}

std::vector<Eigen::Vector3d> filteredCoeffs(
  const std::vector<int> & indices,
  const std::vector<Eigen::Vector3d> & coeffs)
{
  return indices | ranges::views::transform([&](int i) {return coeffs[i];}) | ranges::to_vector;
}

std::vector<Eigen::Vector3d> filteredPoints(
  const std::vector<int> & indices,
  const pcl::PointCloud<pcl::PointXYZ>::Ptr & pointcloud)
{
  const auto f = [&](int i) {return getXYZ(pointcloud->at(i));};
  return indices | ranges::views::transform(f) | ranges::to_vector;
}

std::tuple<std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector3d>, std::vector<double>>
OptimizationProblem::fromEdge(const Eigen::Affine3d & point_to_map) const
{
  std::vector<Eigen::Vector3d> coeffs(edge_scan_->size());
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

    if (u.norm() >= 1.0) {
      continue;
    }

    coeffs[i] = d12.cross(u);
    flags[i] = true;
  }

  const std::vector<int> indices = filteredIndices(flags);
  const std::vector<Eigen::Vector3d> points = filteredPoints(indices, edge_scan_);
  const std::vector<Eigen::Vector3d> coeffs_filtered = filteredCoeffs(indices, coeffs);
  const std::vector<double> b(coeffs_filtered.size(), -1.0);
  return {points, coeffs_filtered, b};
}

std::tuple<std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector3d>, std::vector<double>>
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
    const Eigen::MatrixXd X = get(surface_map_, indices).transpose();
    const Eigen::Vector3d w = solveLinear(X, g);

    if (!validatePlane(X, w)) {
      continue;
    }

    const Eigen::Vector3d q = getXYZ(p);
    const double pd2 = w.dot(q) + 1.0;
    const double norm = w.norm();

    coeffs[i] = w / norm;
    b[i] = -pd2 / norm;
    flags[i] = true;
  }

  const std::vector<int> indices = filteredIndices(flags);
  const std::vector<Eigen::Vector3d> points = filteredPoints(indices, surface_scan_);
  const std::vector<Eigen::Vector3d> coeffs_filtered = filteredCoeffs(indices, coeffs);
  const std::vector<double> b_filtered =
    indices | ranges::views::transform([&](int i) {return b[i];}) | ranges::to_vector;
  return {points, coeffs_filtered, b_filtered};
}

Eigen::MatrixXd makeMatrixA(
  const std::vector<Eigen::Vector3d> & points,
  const std::vector<Eigen::Vector3d> & coeffs,
  const Eigen::Vector3d & rpy)
{
  const Eigen::Matrix3d JX = dRdx(rpy);
  const Eigen::Matrix3d JY = dRdy(rpy);
  const Eigen::Matrix3d JZ = dRdz(rpy);

  Eigen::MatrixXd A(points.size(), 6);
  for (unsigned int i = 0; i < points.size(); i++) {
    // in camera

    const Eigen::Vector3d point = points.at(i);
    const Eigen::Vector3d coeff = coeffs.at(i);

    // lidar -> camera
    A(i, 0) = coeff.dot(JX * point);
    A(i, 1) = coeff.dot(JY * point);
    A(i, 2) = coeff.dot(JZ * point);
    A(i, 3) = coeff(0);
    A(i, 4) = coeff(1);
    A(i, 5) = coeff(2);
  }
  return A;
}

std::tuple<Eigen::MatrixXd, Eigen::VectorXd>
OptimizationProblem::run(const Vector6d & posevec) const
{
  const Eigen::Affine3d point_to_map = getTransformation(posevec);
  const auto [edge_points, edge_coeffs, edge_coeffs_b] = fromEdge(point_to_map);
  const auto [surface_points, surface_coeffs, surface_coeffs_b] = fromSurface(point_to_map);

  const auto points = ranges::views::concat(edge_points, surface_points) | ranges::to_vector;
  const auto coeffs = ranges::views::concat(edge_coeffs, surface_coeffs) | ranges::to_vector;
  auto b_vector = ranges::views::concat(edge_coeffs_b, surface_coeffs_b) | ranges::to_vector;

  assert(points.size() == coeffs.size());
  assert(points.size() == b_vector.size());
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
