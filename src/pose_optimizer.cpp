#include "pose_optimizer.hpp"
#include "jacobian.h"
#include "utility.h"
#include "homogeneous.h"

#include <Eigen/Eigenvalues>

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
  const pcl::PointCloud<PointType>::Ptr & pointcloud,
  const std::vector<int> & indices)
{
  Eigen::Matrix<double, 5, 3> A = Eigen::Matrix<double, 5, 3>::Zero();
  for (int j = 0; j < 5; j++) {
    A.row(j) = getXYZ(pointcloud->at(indices[j]));
  }
  return A;
}

PoseOptimizer::PoseOptimizer(
  const int N_SCAN, const int Horizon_SCAN, const int numberOfCores)
: N_SCAN(N_SCAN), Horizon_SCAN(Horizon_SCAN), numberOfCores(numberOfCores)
{
}

bool PoseOptimizer::LMOptimization(
  const pcl::PointCloud<PointType> & laserCloudOri,
  const pcl::PointCloud<PointType> & coeffSel,
  const int iterCount, bool & isDegenerate, Vector6d & posevec) const
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

std::tuple<pcl::PointCloud<PointType>, pcl::PointCloud<PointType>>
PoseOptimizer::optimization(
  const pcl::PointCloud<PointType> & laserCloudCornerLastDS,
  const pcl::PointCloud<PointType> & laserCloudSurfLastDS,
  const KDTree<PointType> & kdtreeCornerFromMap,
  const KDTree<PointType> & kdtreeSurfFromMap,
  const pcl::PointCloud<PointType>::Ptr & laserCloudCornerFromMapDS,
  const pcl::PointCloud<PointType>::Ptr & laserCloudSurfFromMapDS,
  const Vector6d & posevec) const
{
  const Eigen::Affine3d point_to_map = getTransformation(posevec);
  std::vector<PointType> laserCloudOriCornerVec(N_SCAN * Horizon_SCAN);
  std::vector<PointType> coeffSelCornerVec(N_SCAN * Horizon_SCAN);
  // corner point holder for parallel computation
  std::vector<bool> laserCloudOriCornerFlag(N_SCAN * Horizon_SCAN, false);

  // corner optimization
  #pragma omp parallel for num_threads(numberOfCores)
  for (unsigned int i = 0; i < laserCloudCornerLastDS.size(); i++) {
    const PointType point = laserCloudCornerLastDS.at(i);
    const Eigen::Vector3d map_point = point_to_map * getXYZ(point);
    const PointType p = makePoint(map_point, point.intensity);
    const auto [indices, squared_distances] = kdtreeCornerFromMap.nearestKSearch(p, 5);

    if (squared_distances[4] >= 1.0) {
      continue;
    }

    Eigen::Vector3d c = Eigen::Vector3d::Zero();
    for (int j = 0; j < 5; j++) {
      c += getXYZ(laserCloudCornerFromMapDS->at(indices[j]));
    }
    c /= 5.0;

    Eigen::Matrix3d sa = Eigen::Matrix3d::Zero();

    for (int j = 0; j < 5; j++) {
      const Eigen::Vector3d x = getXYZ(laserCloudCornerFromMapDS->at(indices[j]));
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
    const Eigen::Vector3d p0 = map_point;
    const Eigen::Vector3d p1 = c + 0.1 * v1.row(0).transpose();
    const Eigen::Vector3d p2 = c - 0.1 * v1.row(0).transpose();

    const Eigen::Vector3d d01 = p0 - p1;
    const Eigen::Vector3d d02 = p0 - p2;
    const Eigen::Vector3d d12 = p1 - p2;

    // const Eigen::Vector3d d012(d01(0) * d02(1) - d02(0) * d01(1),
    //                            d01(0) * d02(2) - d02(0) * d01(2),
    //                            d01(1) * d02(2) - d02(1) * d01(2));
    const Eigen::Vector3d cross(
      d01(1) * d02(2) - d01(2) * d02(1),
      d01(2) * d02(0) - d01(0) * d02(2),
      d01(0) * d02(1) - d01(1) * d02(0));

    const double a012 = cross.norm();

    const double l12 = d12.norm();

    // possible bag. maybe the commented one is correct
    // const Eigen::Vector3d v(
    //   (d12(1) * cross(2) - cross(2) * d12(1)),
    //   (d12(2) * cross(0) - cross(0) * d12(2)),
    //   (d12(0) * cross(1) - cross(1) * d12(0)));

    const Eigen::Vector3d v(
      (d12(1) * cross(2) - d12(2) * cross(1)),
      (d12(2) * cross(0) - d12(0) * cross(2)),
      (d12(0) * cross(1) - d12(1) * cross(0)));

    const double ld2 = a012 / l12;

    const double s = 1 - 0.9 * fabs(ld2);

    if (s <= 0.1) {
      continue;
    }
    laserCloudOriCornerVec[i] = point;
    coeffSelCornerVec[i] = makePoint(s * v / (a012 * l12), s * ld2);
    laserCloudOriCornerFlag[i] = true;
  }

  std::vector<PointType> laserCloudOriSurfVec(N_SCAN * Horizon_SCAN);
  std::vector<PointType> coeffSelSurfVec(N_SCAN * Horizon_SCAN);

  // surf point holder for parallel computation
  std::vector<bool> laserCloudOriSurfFlag(N_SCAN * Horizon_SCAN, false);

  // surface optimization
  #pragma omp parallel for num_threads(numberOfCores)
  for (unsigned int i = 0; i < laserCloudSurfLastDS.size(); i++) {
    const PointType point = laserCloudSurfLastDS.at(i);
    const Eigen::Vector3d map_point = point_to_map * getXYZ(point);
    const PointType p = makePoint(map_point, point.intensity);
    const auto [indices, squared_distances] = kdtreeSurfFromMap.nearestKSearch(p, 5);

    if (squared_distances[4] >= 1.0) {
      continue;
    }

    const Eigen::Matrix<double, 5, 1> b = -1.0 * Eigen::Matrix<double, 5, 1>::Ones();
    const Eigen::Matrix<double, 5, 3> A = makeMatrixA(laserCloudSurfFromMapDS, indices);
    const Eigen::Vector3d x = A.colPivHouseholderQr().solve(b);

    if (!validatePlane(A, x)) {
      continue;
    }

    const Eigen::Vector4d y = toHomogeneous(x) / x.norm();
    const Eigen::Vector4d q = toHomogeneous(map_point);
    const float pd2 = y.transpose() * q;
    const float s = 1 - 0.9 * fabs(pd2) / sqrt(map_point.norm());

    if (s <= 0.1) {
      continue;
    }

    laserCloudOriSurfVec[i] = point;
    coeffSelSurfVec[i] = makePoint((s / x.norm()) * x, s * pd2);
    laserCloudOriSurfFlag[i] = true;
  }

  pcl::PointCloud<PointType> laserCloudOri;
  pcl::PointCloud<PointType> coeffSel;

  // combine corner coeffs
  for (unsigned int i = 0; i < laserCloudCornerLastDS.size(); ++i) {
    if (laserCloudOriCornerFlag[i]) {
      laserCloudOri.push_back(laserCloudOriCornerVec[i]);
      coeffSel.push_back(coeffSelCornerVec[i]);
    }
  }
  // combine surf coeffs
  for (unsigned int i = 0; i < laserCloudSurfLastDS.size(); ++i) {
    if (laserCloudOriSurfFlag[i]) {
      laserCloudOri.push_back(laserCloudOriSurfVec[i]);
      coeffSel.push_back(coeffSelSurfVec[i]);
    }
  }

  return {laserCloudOri, coeffSel};
}

std::tuple<Vector6d, bool> PoseOptimizer::run(
  const pcl::PointCloud<PointType> & laserCloudCornerLastDS,
  const pcl::PointCloud<PointType> & laserCloudSurfLastDS,
  const pcl::PointCloud<PointType>::Ptr & laserCloudCornerFromMapDS,
  const pcl::PointCloud<PointType>::Ptr & laserCloudSurfFromMapDS,
  const Vector6d & initial_posevec) const
{
  Vector6d posevec = initial_posevec;

  const KDTree<PointType> kdtreeCornerFromMap(laserCloudCornerFromMapDS);
  const KDTree<PointType> kdtreeSurfFromMap(laserCloudSurfFromMapDS);

  bool isDegenerate = false;
  for (int iterCount = 0; iterCount < 30; iterCount++) {
    const auto [laserCloudOri, coeffSel] = optimization(
      laserCloudCornerLastDS, laserCloudSurfLastDS,
      kdtreeCornerFromMap, kdtreeSurfFromMap,
      laserCloudCornerFromMapDS, laserCloudSurfFromMapDS,
      posevec
    );

    if (LMOptimization(laserCloudOri, coeffSel, iterCount, isDegenerate, posevec)) {
      break;
    }
  }
  return {posevec, isDegenerate};
}
