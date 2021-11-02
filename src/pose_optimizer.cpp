#include "pose_optimizer.hpp"
#include "jacobian.h"
#include "utility.hpp"

#include <Eigen/Eigenvalues>

bool LMOptimization(
  const pcl::PointCloud<pcl::PointXYZ> & points,
  const std::vector<Eigen::Vector4d> & coeffs,
  const int iterCount, bool & isDegenerate, Vector6d & posevec)
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
  if (points.size() < 50) {
    return false;
  }

  Eigen::MatrixXd A(points.size(), 6);
  Eigen::VectorXd b(points.size());

  for (unsigned int i = 0; i < points.size(); i++) {
    const Eigen::Vector4d c = coeffs.at(i);
    const float intensity = c(3);

    // in camera

    const Eigen::Vector3d point_ori(
      points.at(i).y,
      points.at(i).z,
      points.at(i).x);

    const Eigen::Vector3d coeff_vec(c(1), c(2), c(0));

    const Eigen::Matrix3d MX = dRdx(posevec(0), posevec(2), posevec(1));
    const float arx = (MX * point_ori).dot(coeff_vec);

    const Eigen::Matrix3d MY = dRdy(posevec(0), posevec(2), posevec(1));
    const float ary = (MY * point_ori).dot(coeff_vec);

    const Eigen::Matrix3d MZ = dRdz(posevec(0), posevec(2), posevec(1));
    const float arz = (MZ * point_ori).dot(coeff_vec);

    // lidar -> camera
    A(i, 0) = arz;
    A(i, 1) = arx;
    A(i, 2) = ary;
    A(i, 3) = c(0);
    A(i, 4) = c(1);
    A(i, 5) = c(2);
    b(i) = -intensity;
  }

  const Eigen::MatrixXd AtA = A.transpose() * A;
  const Eigen::VectorXd AtB = A.transpose() * b;

  const Eigen::VectorXd matX = AtA.householderQr().solve(AtB);

  if (iterCount == 0) {
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(AtA);
    const Eigen::VectorXd eigenvalues = es.eigenvalues();

    isDegenerate = (eigenvalues.array() < 100.0).any();
  }

  if (!isDegenerate) {
    posevec += matX;
  }

  const float deltaR = rad2deg(matX.head(3)).norm();
  const float deltaT = (100 * matX.tail(3)).norm();

  if (deltaR < 0.05 && deltaT < 0.05) {
    return true; // converged
  }
  return false; // keep optimizing
}

std::tuple<Vector6d, bool> optimizePose(
  const CloudOptimizer & cloud_optimizer,
  const Vector6d & initial_posevec)
{
  Vector6d posevec = initial_posevec;

  bool isDegenerate = false;
  for (int iterCount = 0; iterCount < 30; iterCount++) {
    const auto [points, coeffs] = cloud_optimizer.run(posevec);

    if (LMOptimization(points, coeffs, iterCount, isDegenerate, posevec)) {
      break;
    }
  }
  return {posevec, isDegenerate};
}
