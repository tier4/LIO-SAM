#include "pose_optimizer.hpp"
#include "jacobian.h"
#include "utility.hpp"

#include <Eigen/Eigenvalues>

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

bool LMOptimization(
  const std::vector<Eigen::Vector3d> & points,
  const std::vector<Eigen::Vector3d> & coeffs,
  const Eigen::VectorXd & b,
  const int iter, const bool & is_degenerate, Vector6d & posevec)
{
  // This optimization is from the original loam_velodyne by Ji Zhang,
  // need to cope with coordinate transformation
  // lidar <- camera      ---     camera <- lidar
  // x = z                ---     x = y
  // y = x                ---     y = z
  // z = y                ---     z = x
  // roll = yaw           ---     roll = pitch
  // pitch = roll         ---     pitch = yaw
  // yaw = pitch          ---     yaw = roll

  // lidar -> camera

  const Eigen::MatrixXd A = makeMatrixA(points, coeffs, posevec.head(3));

  const Eigen::MatrixXd AtA = A.transpose() * A;
  const Eigen::VectorXd AtB = A.transpose() * b;

  const Eigen::VectorXd dx = solveLinear(AtA, AtB);

  if (!is_degenerate) {
    posevec += dx;
  }

  return checkConvergence(dx);
}

std::tuple<Vector6d, bool> optimizePose(
  const CloudOptimizer & cloud_optimizer,
  const Vector6d & initial_posevec)
{
  const bool is_degenerate = isDegenerate(cloud_optimizer, initial_posevec);

  Vector6d posevec = initial_posevec;
  for (int iter = 0; iter < 30; iter++) {
    const auto [points, coeffs, b_vector] = cloud_optimizer.run(posevec);
    if (points.size() < 50) {
      continue;
    }

    const Eigen::Map<const Eigen::VectorXd> b(b_vector.data(), b_vector.size());
    if (LMOptimization(points, coeffs, b, iter, is_degenerate, posevec)) {
      break;
    }
  }
  return {posevec, is_degenerate};
}
