#include "utility.h"

tf::Pose poseMsgToTF(const geometry_msgs::Pose & msg)
{
  tf::Pose pose;
  tf::poseMsgToTF(msg, pose);
  return pose;
}

tf::Quaternion rpyToTfQuaternion(const Eigen::Vector3d & rpy)
{
  tf::Quaternion q;
  q.setRPY(rpy(0), rpy(1), rpy(2));
  return q;
}

Eigen::Vector3d getRPY(const tf::Quaternion & q)
{
  double roll, pitch, yaw;
  tf::Matrix3x3(q).getRPY(roll, pitch, yaw);
  return Eigen::Vector3d(roll, pitch, yaw);
}

tf::Quaternion interpolate(
  const tf::Quaternion & q0, const tf::Quaternion & q1,
  const tfScalar weight)
{
  return q0.slerp(q1, weight);
}

Eigen::Vector3d interpolate(
  const Eigen::Vector3d & rpy0, const Eigen::Vector3d & rpy1, const tfScalar weight)
{
  const tf::Quaternion q0 = rpyToTfQuaternion(rpy0);
  const tf::Quaternion q1 = rpyToTfQuaternion(rpy1);
  return getRPY(interpolate(q0, q1, weight));
}

geometry_msgs::Vector3 pointToVector3(const geometry_msgs::Point & p)
{
  geometry_msgs::Vector3 v;
  v.x = p.x;
  v.y = p.y;
  v.z = p.z;
  return v;
}

geometry_msgs::Point vector3ToPoint(const geometry_msgs::Vector3 & v)
{
  geometry_msgs::Point p;
  p.x = v.x;
  p.y = v.y;
  p.z = v.z;
  return p;
}

geometry_msgs::Transform poseToTransform(const geometry_msgs::Pose & pose)
{
  geometry_msgs::Transform transform;
  transform.translation = pointToVector3(pose.position);
  transform.rotation = pose.orientation;
  return transform;
}

geometry_msgs::Pose transformToPose(const geometry_msgs::Transform & transform)
{
  geometry_msgs::Pose pose;
  pose.position = vector3ToPoint(transform.translation);
  pose.orientation = transform.rotation;
  return pose;
}

geometry_msgs::TransformStamped poseToTransform(
  const geometry_msgs::PoseStamped & pose,
  const std::string & child_frame_id)
{
  geometry_msgs::TransformStamped transform;
  transform.transform = poseToTransform(pose.pose);
  transform.header = pose.header;
  transform.child_frame_id = child_frame_id;
  return transform;
}

sensor_msgs::PointCloud2 toRosMsg(const pcl::PointCloud<PointType> & pointcloud)
{
  sensor_msgs::PointCloud2 msg;
  pcl::toROSMsg(pointcloud, msg);
  return msg;
}

sensor_msgs::PointCloud2 toRosMsg(
  const pcl::PointCloud<PointType> & pointcloud,
  const ros::Time stamp,
  const std::string frame)
{
  sensor_msgs::PointCloud2 msg = toRosMsg(pointcloud);
  msg.header.stamp = stamp;
  msg.header.frame_id = frame;
  return msg;
}

nav_msgs::Odometry makeOdometry(
  const ros::Time & timestamp,
  const std::string & frame_id,
  const std::string & child_frame_id,
  const geometry_msgs::Pose & pose)
{
  nav_msgs::Odometry odometry;
  odometry.header.stamp = timestamp;
  odometry.header.frame_id = frame_id;
  odometry.child_frame_id = child_frame_id;
  odometry.pose.pose = pose;
  return odometry;
}

PointType makePoint(const Eigen::Vector3d & point, const float intensity)
{
  const Eigen::Vector3f q = point.cast<float>();
  PointType p;
  p.x = q(0);
  p.y = q(1);
  p.z = q(2);
  p.intensity = intensity;
  return p;
}

Eigen::Affine3d makeAffine(const Eigen::Vector3d & rpy, const Eigen::Vector3d & point)
{
  Eigen::Affine3d transform;
  pcl::getTransformation(point(0), point(1), point(2), rpy(0), rpy(1), rpy(2), transform);
  return transform;
}

Eigen::Affine3d makeAffine(
  const geometry_msgs::Vector3 & rpy,
  const geometry_msgs::Vector3 & point)
{
  Eigen::Affine3d transform;
  pcl::getTransformation(point.x, point.y, point.z, rpy.x, rpy.y, rpy.z, transform);
  return transform;
}

Eigen::Vector3d quaternionToRPY(const tf::Quaternion & orientation)
{
  Eigen::Vector3d rpy;
  tf::Matrix3x3(orientation).getRPY(rpy(0), rpy(1), rpy(2));
  return rpy;
}

Eigen::Vector3d quaternionToRPY(const geometry_msgs::Quaternion & orientation)
{
  tf::Quaternion quat;
  tf::quaternionMsgToTF(orientation, quat);
  return quaternionToRPY(quat);
}

Eigen::Quaterniond quaternionToEigen(const geometry_msgs::Quaternion & quat_msg)
{
  Eigen::Quaterniond quat_eigen;
  tf::quaternionMsgToEigen(quat_msg, quat_eigen);
  return quat_eigen;
}

geometry_msgs::Quaternion eigenToQuaternion(const Eigen::Quaterniond & quat_eigen)
{
  geometry_msgs::Quaternion quat_msg;
  tf::quaternionEigenToMsg(quat_eigen, quat_msg);
  return quat_msg;
}

Eigen::Affine3d transformToAffine(const geometry_msgs::Transform & transform)
{
  Eigen::Affine3d affine;
  tf::transformMsgToEigen(transform, affine);
  return affine;
}

Eigen::Affine3d poseToAffine(const geometry_msgs::Pose & pose)
{
  Eigen::Affine3d affine;
  tf::poseMsgToEigen(pose, affine);
  return affine;
}

geometry_msgs::Vector3 eigenToVector3(const Eigen::Vector3d & v)
{
  geometry_msgs::Vector3 p;
  p.x = v[0];
  p.y = v[1];
  p.z = v[2];
  return p;
}

std::tuple<Eigen::Vector3d, Eigen::Vector3d> getXYZRPY(const Eigen::Affine3d & affine)
{
  double x, y, z, roll, pitch, yaw;
  pcl::getTranslationAndEulerAngles(affine, x, y, z, roll, pitch, yaw);
  return {Eigen::Vector3d(x, y, z), Eigen::Vector3d(roll, pitch, yaw)};
}

geometry_msgs::Point eigenToPoint(const Eigen::Vector3d & v)
{
  geometry_msgs::Point p;
  p.x = v[0];
  p.y = v[1];
  p.z = v[2];
  return p;
}

geometry_msgs::Pose makePose(
  const geometry_msgs::Quaternion & orientation,
  const geometry_msgs::Point & position)
{
  geometry_msgs::Pose pose;
  pose.position = position;
  pose.orientation = orientation;
  return pose;
}

Eigen::Affine3d getTransformation(const Vector6d & posevec)
{
  Eigen::Affine3d transform;
  pcl::getTransformation(
    posevec(3), posevec(4), posevec(5),
    posevec(0), posevec(1), posevec(2), transform);
  return transform;
}

geometry_msgs::PoseStamped makePoseStamped(
  const geometry_msgs::Pose & pose,
  const std::string & frame_id,
  const double time)
{
  geometry_msgs::PoseStamped pose_stamped;
  pose_stamped.header.stamp = ros::Time().fromSec(time);
  pose_stamped.header.frame_id = frame_id;
  pose_stamped.pose = pose;
  return pose_stamped;
}

geometry_msgs::Pose makePose(const gtsam::Pose3 & pose)
{
  const auto q = eigenToQuaternion(pose.rotation().toQuaternion());
  const auto p = eigenToPoint(pose.translation());
  return makePose(q, p);
}

geometry_msgs::TransformStamped makeTransformStamped(
  const ros::Time & timestamp,
  const std::string & frame_id,
  const std::string & child_frame_id,
  const geometry_msgs::Transform & transform)
{
  geometry_msgs::TransformStamped transform_stamped;
  transform_stamped.header.stamp = timestamp;
  transform_stamped.header.frame_id = frame_id;
  transform_stamped.child_frame_id = child_frame_id;
  transform_stamped.transform = transform;
  return transform_stamped;
}

geometry_msgs::Transform makeTransform(
  const geometry_msgs::Quaternion & rotation,
  const geometry_msgs::Vector3 & translation)
{
  geometry_msgs::Transform transform;
  transform.rotation = rotation;
  transform.translation = translation;
  return transform;
}

geometry_msgs::Transform makeTransform(const gtsam::Pose3 & pose)
{
  const auto q = eigenToQuaternion(pose.rotation().toQuaternion());
  const auto p = eigenToVector3(pose.translation());
  return makeTransform(q, p);
}

geometry_msgs::Pose makePose(const Eigen::Vector3d & rpy, const Eigen::Vector3d & xyz)
{
  const auto orientation = tf::createQuaternionMsgFromRollPitchYaw(rpy(0), rpy(1), rpy(2));
  const auto position = eigenToPoint(xyz);
  return makePose(orientation, position);
}

geometry_msgs::Pose makePose(const Vector6d & posevec)
{
  return makePose(posevec.head(3), posevec.tail(3));
}

geometry_msgs::Twist makeTwist(
  const geometry_msgs::Vector3 & angular,
  const geometry_msgs::Vector3 & linear)
{
  geometry_msgs::Twist twist;
  twist.angular = angular;
  twist.linear = linear;
  return twist;
}

geometry_msgs::Pose affineToPose(const Eigen::Affine3d & affine)
{
  geometry_msgs::Pose pose;
  tf::poseEigenToMsg(affine, pose);
  return pose;
}

IMUConverter::IMUConverter()
{
  std::vector<double> extRotV;
  std::vector<double> extRPYV;
  nh.param<std::vector<double>>("lio_sam/extrinsicRot", extRotV, std::vector<double>());
  nh.param<std::vector<double>>("lio_sam/extrinsicRPY", extRPYV, std::vector<double>());
  extRot = Eigen::Map<const RowMajorMatrixXd>(extRotV.data(), 3, 3);
  Eigen::Matrix3d extRPY = Eigen::Map<const RowMajorMatrixXd>(extRPYV.data(), 3, 3);
  extQRPY = Eigen::Quaterniond(extRPY);
}

sensor_msgs::Imu IMUConverter::imuConverter(const sensor_msgs::Imu & imu_in) const
{
  sensor_msgs::Imu imu_out = imu_in;
  // rotate acceleration
  const Eigen::Vector3d acc = vector3ToEigen(imu_in.linear_acceleration);
  imu_out.linear_acceleration = eigenToVector3(extRot * acc);

  const Eigen::Vector3d gyr = vector3ToEigen(imu_in.angular_velocity);
  imu_out.angular_velocity = eigenToVector3(extRot * gyr);

  const Eigen::Quaterniond q_from = quaternionToEigen(imu_in.orientation);
  const Eigen::Quaterniond q_final = q_from * extQRPY;
  imu_out.orientation = eigenToQuaternion(q_final);

  if (q_final.norm() < 0.1) {
    throw std::runtime_error("Invalid quaternion, please use a 9-axis IMU!");
  }

  return imu_out;
}
