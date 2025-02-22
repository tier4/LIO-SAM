#ifndef TRANSFORM_FUSION_HPP_
#define TRANSFORM_FUSION_HPP_

tf::Transform getLidarToBaseLink(
  const std::string & lidarFrame,
  const std::string & baselinkFrame)
{
  if (lidarFrame == baselinkFrame) {
    return identityTransform();
  }

  tf::StampedTransform transform;
  try {
    tf::TransformListener listener;
    listener.lookupTransform(lidarFrame, baselinkFrame, ros::Time(0), transform);
  } catch (tf::TransformException ex) {
    ROS_ERROR("%s", ex.what());
  }

  return transform;
}

class OdomToBaselink
{
private:
  const tf::Transform lidar_to_baselink;
  const std::string odometryFrame;
  const std::string baselinkFrame;

public:
  OdomToBaselink(
    const std::string & lidarFrame,
    const std::string & odometryFrame,
    const std::string & baselinkFrame)
  : lidar_to_baselink(getLidarToBaseLink(lidarFrame, baselinkFrame)),
    odometryFrame(odometryFrame),
    baselinkFrame(baselinkFrame)
  {
  }

  tf::StampedTransform get(
    const geometry_msgs::Pose & odometry,
    const ros::Time & timestamp) const
  {
    const tf::Transform lidar_odometry = poseMsgToTF(odometry);
    return tf::StampedTransform(
      lidar_odometry * lidar_to_baselink,
      timestamp, odometryFrame, baselinkFrame);
  }
};

class ImuPath
{
private:
  nav_msgs::Path imuPath;

  const std::string odometryFrame;

public:
  ImuPath(const std::string odometryFrame)
  : odometryFrame(odometryFrame) {}

  nav_msgs::Path make(
    const ros::Time & timestamp,
    const geometry_msgs::Pose & pose,
    const double lidar_odometry_time)
  {
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header.stamp = timestamp;
    pose_stamped.header.frame_id = odometryFrame;
    pose_stamped.pose = pose;
    imuPath.poses.push_back(pose_stamped);
    while (
      !imuPath.poses.empty() &&
      imuPath.poses.front().header.stamp.toSec() < lidar_odometry_time - 1.0)
    {
      imuPath.poses.erase(imuPath.poses.begin());
    }
    imuPath.header.stamp = timestamp;
    imuPath.header.frame_id = odometryFrame;

    return imuPath;
  }
};

Eigen::Affine3d latestOdometry(
  const geometry_msgs::Transform & front_transform,
  const geometry_msgs::Transform & back_transform,
  const geometry_msgs::Transform & lidar_odom)
{
  const Eigen::Affine3d front = transformToAffine(front_transform);
  const Eigen::Affine3d back = transformToAffine(back_transform);
  const Eigen::Affine3d incre = front.inverse() * back;
  return transformToAffine(lidar_odom) * incre;
}

class TransformFusion : public ParamServer
{
public:
  std::mutex mtx;

  const ros::Subscriber subLaserOdometry;
  const ros::Subscriber subImuOdometry;

  const ros::Publisher pubImuPath;

  geometry_msgs::Transform lidar_odom;

  const tf::Transform lidar_to_baselink;
  const OdomToBaselink odom_to_baselink;
  tf::TransformBroadcaster broadcaster;

  double lidar_odometry_time;
  std::deque<geometry_msgs::TransformStamped> odometry_queue_;

  tf::TransformBroadcaster tf_map_to_odom;

  ImuPath imu_path;

  TransformFusion()
  : subLaserOdometry(nh.subscribe<nav_msgs::Odometry>(
        "lio_sam/mapping/odometry",
        5, &TransformFusion::lidarOdometryHandler, this,
        ros::TransportHints().tcpNoDelay())),
    subImuOdometry(nh.subscribe<geometry_msgs::TransformStamped>(
        imu_incremental_odometry_topic,
        2000, &TransformFusion::imuOdometryHandler, this,
        ros::TransportHints().tcpNoDelay())),
    pubImuPath(nh.advertise<nav_msgs::Path>("lio_sam/imu/path", 1)),
    odom_to_baselink(OdomToBaselink(lidarFrame, odometryFrame, baselinkFrame)),
    lidar_odometry_time(-1.0),
    imu_path(ImuPath(odometryFrame))
  {
  }

  void lidarOdometryHandler(const nav_msgs::Odometry::ConstPtr & odom_msg)
  {
    std::lock_guard<std::mutex> lock(mtx);

    lidar_odom = poseToTransform(odom_msg->pose.pose);

    lidar_odometry_time = odom_msg->header.stamp.toSec();
  }

  void imuOdometryHandler(const geometry_msgs::TransformStamped::ConstPtr & odom_msg)
  {
    const auto stamp = odom_msg->header.stamp;

    tf_map_to_odom.sendTransform(
      tf::StampedTransform(identityTransform(), stamp, mapFrame, odometryFrame));

    std::lock_guard<std::mutex> lock(mtx);

    odometry_queue_.push_back(*odom_msg);

    if (lidar_odometry_time == -1) {
      return;
    }

    dropBefore(lidar_odometry_time, odometry_queue_);
    const auto front = odometry_queue_.front().transform;
    const auto back = odom_msg->transform;
    const auto pose = affineToPose(latestOdometry(front, back, lidar_odom));

    broadcaster.sendTransform(odom_to_baselink.get(pose, stamp));

    if (pubImuPath.getNumSubscribers() == 0) {
      return;
    }
    pubImuPath.publish(imu_path.make(stamp, pose, lidar_odometry_time));
  }
};

#endif
