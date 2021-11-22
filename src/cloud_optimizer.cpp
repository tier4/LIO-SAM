#include "cloud_optimizer.hpp"
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
CloudOptimizer::fromEdge(const Eigen::Affine3d & point_to_map) const
{
  std::vector<Eigen::Vector3d> coeffs(edge_->size());
  std::vector<double> b(edge_->size());
  std::vector<bool> flags(edge_->size(), false);

  #pragma omp parallel for num_threads(numberOfCores)
  for (unsigned int i = 0; i < edge_->size(); i++) {
    const pcl::PointXYZ p = transform(point_to_map, edge_->at(i));
    const auto [indices, squared_distances] = edge_kdtree_.nearestKSearch(p, n_neighbors);

    if (squared_distances.back() >= 1.0) {
      continue;
    }

    const Eigen::MatrixXd neighbors = get(edge_map_, indices);
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(calcCovariance(neighbors));
    const Eigen::Vector3d eigenvalues = solver.eigenvalues();
    const Eigen::Vector3d eigenvector = solver.eigenvectors().col(0);

    if (eigenvalues(0) <= 3 * eigenvalues(1)) {
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
    const double a012 = u.norm();
    const double l12 = d12.norm();

    const double k = fabs(a012 / l12);
    if (k >= 1.0) {
      continue;
    }

    const double s = 1 - 0.9 * k;
    coeffs[i] = (s / (l12 * a012)) * v;
    b[i] = -(s / l12) * a012;
    flags[i] = true;
  }
  return {coeffs, b, flags};
}

std::tuple<std::vector<Eigen::Vector3d>, std::vector<double>, std::vector<bool>>
CloudOptimizer::fromSurface(const Eigen::Affine3d & point_to_map) const
{
  std::vector<Eigen::Vector3d> coeffs(surface_->size());
  std::vector<double> b(surface_->size());
  std::vector<bool> flags(surface_->size(), false);

  // surface optimization
  #pragma omp parallel for num_threads(numberOfCores)
  for (unsigned int i = 0; i < surface_->size(); i++) {
    const pcl::PointXYZ p = transform(point_to_map, surface_->at(i));
    const auto [indices, squared_distances] = surface_kdtree_.nearestKSearch(p, 5);

    if (squared_distances[4] >= 1.0) {
      continue;
    }

    const Eigen::Matrix<double, 5, 1> g = -1.0 * Eigen::Matrix<double, 5, 1>::Ones();
    const Eigen::Matrix<double, 5, 3> A = makeMatrixA(surface_map_, indices);
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

std::tuple<std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector3d>, std::vector<double>>
CloudOptimizer::run(const Vector6d & posevec) const
{
  const Eigen::Affine3d point_to_map = getTransformation(posevec);
  const auto [edge_coeffs, edge_coeffs_b, edge_flags] = fromEdge(point_to_map);
  const auto [surface_coeffs, surface_coeffs_b, surface_flags] = fromSurface(point_to_map);

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

Eigen::MatrixXd makeMatrixA(
  const std::vector<Eigen::Vector3d> & points,
  const std::vector<Eigen::Vector3d> & coeffs,
  const Eigen::Vector3d & rpy)
{
  Eigen::MatrixXd A(points.size(), 6);
  for (unsigned int i = 0; i < points.size(); i++) {
    // in camera

    const Eigen::Vector3d p = points.at(i);
    const Eigen::Vector3d c = coeffs.at(i);
    const Eigen::Vector3d point_ori(p(1), p(2), p(0));
    const Eigen::Vector3d coeff_vec(c(1), c(2), c(0));

    const Eigen::Matrix3d MX = dRdx(rpy(0), rpy(2), rpy(1));
    const float arx = (MX * point_ori).dot(coeff_vec);

    const Eigen::Matrix3d MY = dRdy(rpy(0), rpy(2), rpy(1));
    const float ary = (MY * point_ori).dot(coeff_vec);

    const Eigen::Matrix3d MZ = dRdz(rpy(0), rpy(2), rpy(1));
    const float arz = (MZ * point_ori).dot(coeff_vec);

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

bool isDegenerate(const CloudOptimizer & cloud_optimizer, const Vector6d & posevec)
{
  const auto [points, coeffs, b_vector] = cloud_optimizer.run(posevec);
  const Eigen::MatrixXd A = makeMatrixA(points, coeffs, posevec.head(3));
  const Eigen::MatrixXd AtA = A.transpose() * A;
  const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(AtA);
  const Eigen::VectorXd eigenvalues = es.eigenvalues();
  return (eigenvalues.array() < 100.0).any();
}

Eigen::VectorXd calcUpdate(
  const std::vector<Eigen::Vector3d> & points,
  const std::vector<Eigen::Vector3d> & coeffs,
  const Eigen::VectorXd & b,
  const Eigen::Vector3d & rpy)
{
  const Eigen::MatrixXd A = makeMatrixA(points, coeffs, rpy);

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

Vector6d optimizePose(const CloudOptimizer & cloud_optimizer, const Vector6d & initial_posevec)
{
  Vector6d posevec = initial_posevec;
  for (int iter = 0; iter < 30; iter++) {
    const auto [points, coeffs, b_vector] = cloud_optimizer.run(posevec);
    if (points.size() < 50) {
      continue;
    }

    const Eigen::Map<const Eigen::VectorXd> b(b_vector.data(), b_vector.size());
    const Eigen::VectorXd dx = calcUpdate(points, coeffs, b, posevec.head(3));

    posevec += dx;

    if (checkConvergence(dx)) {
      break;
    }
  }
  return posevec;
}
