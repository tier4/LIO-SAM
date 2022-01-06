#include "optimization_problem.hpp"
#include "homogeneous.h"
#include "utility.hpp"
#include "jacobian.h"

#include <Eigen/Eigenvalues>

#include <range/v3/all.hpp>

double pointPlaneDistance(const Eigen::Vector3d & w, const Eigen::Vector3d & x)
{
  return std::abs(w.dot(x) + 1.0) / w.norm();
}

bool validatePlane(const Eigen::MatrixXd & X, const Eigen::Vector3d & w)
{
  for (int j = 0; j < X.rows(); j++) {
    const Eigen::Vector3d x = X.row(j);
    if (pointPlaneDistance(w, x) > 0.2) {
      return false;
    }
  }
  return true;
}

Eigen::MatrixXd get(
  const pcl::PointCloud<pcl::PointXYZ>::Ptr & pointcloud,
  const std::vector<int> & indices)
{
  Eigen::MatrixXd A(indices.size(), 3);
  for (const auto & [j, index] : ranges::views::enumerate(indices)) {
    const Eigen::Vector3d p = getXYZ(pointcloud->at(index));
    A.row(j) = p.transpose();
  }
  return A;
}

Eigen::Matrix3d calcCovariance(const Eigen::MatrixXd & X)
{
  const Eigen::Vector3d c = X.colwise().mean();
  const Eigen::MatrixXd D = X.rowwise() - c.transpose();
  return D.transpose() * D / X.rows();
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

std::vector<int> trueIndices(const std::vector<bool> & flags)
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
  // f(dx) \approx f(0) + J * dx + dx^T * H * dx
  // dx can be obtained by solving H * dx = -J

  std::vector<Eigen::Vector3d> coeffs(edge_scan_->size());
  std::vector<bool> flags(edge_scan_->size(), false);

  #pragma omp parallel for num_threads(n_threads_)
  for (unsigned int i = 0; i < edge_scan_->size(); i++) {
    const pcl::PointXYZ p = transform(point_to_map, edge_scan_->at(i));
    const auto [indices, squared_distances] = edge_kdtree_.nearestKSearch(p, n_neighbors);
    if (squared_distances.back() >= 1.0) {
      continue;
    }

    const Eigen::Matrix<double, n_neighbors, 3> neighbors = get(edge_map_, indices);
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(calcCovariance(neighbors));
    const Eigen::Vector3d eigenvalues = solver.eigenvalues();
    const Eigen::Vector3d eigenvector = solver.eigenvectors().col(2);

    if (eigenvalues(2) <= 3 * eigenvalues(1)) {
      continue;
    }

    const Eigen::Vector3d c = neighbors.colwise().mean();
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

  const std::vector<int> indices = trueIndices(flags);
  const std::vector<Eigen::Vector3d> points = filteredPoints(indices, edge_scan_);
  const std::vector<Eigen::Vector3d> coeffs_filtered = filteredCoeffs(indices, coeffs);
  const std::vector<double> b(coeffs_filtered.size(), -1.0);
  return {points, coeffs_filtered, b};
}

Eigen::Vector3d estimatePlaneCoefficients(const Eigen::MatrixXd & X)
{
  const Eigen::VectorXd g = -1.0 * Eigen::VectorXd::Ones(X.rows());
  return solveLinear(X, g);
}

std::tuple<std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector3d>, std::vector<double>>
OptimizationProblem::fromSurface(const Eigen::Affine3d & point_to_map) const
{
  std::vector<Eigen::Vector3d> coeffs(surface_scan_->size());
  std::vector<double> b(surface_scan_->size());
  std::vector<bool> flags(surface_scan_->size(), false);

  // surface optimization
  #pragma omp parallel for num_threads(n_threads_)
  for (unsigned int i = 0; i < surface_scan_->size(); i++) {
    const pcl::PointXYZ p = transform(point_to_map, surface_scan_->at(i));
    const auto [indices, squared_distances] = surface_kdtree_.nearestKSearch(p, n_neighbors);

    if (squared_distances.back() >= 1.0) {
      continue;
    }

    const Eigen::MatrixXd X = get(surface_map_, indices);
    const Eigen::Vector3d w = estimatePlaneCoefficients(X);

    if (!validatePlane(X, w)) {
      continue;
    }

    const Eigen::Vector3d q = getXYZ(p);
    const double norm = w.norm();

    coeffs[i] = w / norm;
    b[i] = -(w.dot(q) + 1.0) / norm;
    flags[i] = true;
  }

  const std::vector<int> indices = trueIndices(flags);
  const std::vector<Eigen::Vector3d> points = filteredPoints(indices, surface_scan_);
  const std::vector<Eigen::Vector3d> coeffs_filtered = filteredCoeffs(indices, coeffs);
  const std::vector<double> b_filtered =
    indices | ranges::views::transform([&](int i) {return b[i];}) | ranges::to_vector;
  return {points, coeffs_filtered, b_filtered};
}

Eigen::MatrixXd makeJacobian(
  const std::vector<Eigen::Vector3d> & points,
  const std::vector<Eigen::Vector3d> & coeffs,
  const Eigen::Vector3d & rpy)
{
  const Eigen::Matrix3d JX = dRdx(rpy);
  const Eigen::Matrix3d JY = dRdy(rpy);
  const Eigen::Matrix3d JZ = dRdz(rpy);

  Eigen::MatrixXd J(points.size(), 6);
  for (unsigned int i = 0; i < points.size(); i++) {
    // in camera

    const Eigen::Vector3d point = points.at(i);
    const Eigen::Vector3d coeff = coeffs.at(i);

    const Eigen::Vector3d drpdx = JX * point;
    const Eigen::Vector3d drpdy = JY * point;
    const Eigen::Vector3d drpdz = JZ * point;

    // lidar -> camera
    J(i, 0) = coeff.dot(drpdx);  // d ||residual||^2 / d roll
    J(i, 1) = coeff.dot(drpdy);  // d ||residual||^2 / d pitch
    J(i, 2) = coeff.dot(drpdz);  // d ||residual||^2 / d yaw
    J(i, 3) = coeff(0);          // d ||residual||^2 / d tx
    J(i, 4) = coeff(1);          // d ||residual||^2 / d ty
    J(i, 5) = coeff(2);          // d ||residual||^2 / d tz
  }
  return J;
}

std::tuple<Eigen::MatrixXd, Eigen::VectorXd>
OptimizationProblem::make(const Vector6d & posevec) const
{
  const Eigen::Affine3d point_to_map = getTransformation(posevec);
  const auto [edge_points, edge_coeffs, edge_coeffs_b] = fromEdge(point_to_map);
  const auto [surface_points, surface_coeffs, surface_coeffs_b] = fromSurface(point_to_map);

  const auto points = ranges::views::concat(edge_points, surface_points) | ranges::to_vector;
  const auto coeffs = ranges::views::concat(edge_coeffs, surface_coeffs) | ranges::to_vector;
  auto b_vector = ranges::views::concat(edge_coeffs_b, surface_coeffs_b) | ranges::to_vector;

  assert(points.size() == coeffs.size());
  assert(points.size() == b_vector.size());
  const Eigen::MatrixXd J = makeJacobian(points, coeffs, posevec.head(3));
  const Eigen::Map<Eigen::VectorXd> b(b_vector.data(), b_vector.size());
  return {J, b};
}

bool isDegenerate(const OptimizationProblem & problem, const Vector6d & posevec)
{
  const auto [J, b] = problem.make(posevec);
  const Eigen::MatrixXd JtJ = J.transpose() * J;
  const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(JtJ);
  const Eigen::VectorXd eigenvalues = es.eigenvalues();
  return (eigenvalues.array() < 100.0).any();
}

Eigen::VectorXd calcUpdate(const Eigen::MatrixXd & J, const Eigen::VectorXd & b)
{
  const Eigen::MatrixXd JtJ = J.transpose() * J;
  const Eigen::VectorXd JtB = J.transpose() * b;
  return solveLinear(JtJ, JtB);
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
    const auto [J, b] = problem.make(posevec);
    if (J.rows() < 50) {
      continue;
    }

    const Eigen::VectorXd dx = calcUpdate(J, b);

    posevec += dx;

    if (checkConvergence(dx)) {
      break;
    }
  }
  return posevec;
}
