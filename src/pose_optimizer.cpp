#include "pose_optimizer.hpp"
#include "jacobian.h"
#include "utility.h"

#include <Eigen/Eigenvalues>

bool LMOptimization(
  const pcl::PointCloud<PointType> & laserCloudOri,
  const pcl::PointCloud<PointType> & coeffSel,
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
  int laserCloudSelNum = laserCloudOri.size();
  if (laserCloudSelNum < 50) {
    return false;
  }

  Eigen::MatrixXd A(laserCloudSelNum, 6);
  Eigen::VectorXd b(laserCloudSelNum);

  for (int i = 0; i < laserCloudSelNum; i++) {
    // lidar -> camera
    const float intensity = coeffSel.at(i).intensity;

    // in camera

    const Eigen::Vector3d point_ori(
      laserCloudOri.at(i).y,
      laserCloudOri.at(i).z,
      laserCloudOri.at(i).x);

    const Eigen::Vector3d coeff_vec(
      coeffSel.at(i).y,
      coeffSel.at(i).z,
      coeffSel.at(i).x);

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
    A(i, 3) = coeffSel.at(i).x;
    A(i, 4) = coeffSel.at(i).y;
    A(i, 5) = coeffSel.at(i).z;
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
    const auto [laserCloudOri, coeffSel] = cloud_optimizer.run(posevec);

    if (LMOptimization(laserCloudOri, coeffSel, iterCount, isDegenerate, posevec)) {
      break;
    }
  }
  return {posevec, isDegenerate};
}
