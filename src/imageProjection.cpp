#include "utility.h"
#include "lio_sam/cloud_info.h"

struct VelodynePointXYZIRT
{
  PCL_ADD_POINT4D PCL_ADD_INTENSITY;
  uint16_t ring;
  float time;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(
  VelodynePointXYZIRT,
  (float, x, x)(float, y, y) (float, z, z) (float, intensity, intensity)(
    uint16_t, ring,
    ring) (float, time, time)
)

struct OusterPointXYZIRT
{
  PCL_ADD_POINT4D;
  float intensity;
  uint32_t t;
  uint16_t reflectivity;
  uint8_t ring;
  uint16_t noise;
  uint32_t range;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(
  OusterPointXYZIRT,
  (float, x, x)(float, y, y) (float, z, z) (float, intensity, intensity)(uint32_t, t, t) (
    uint16_t,
    reflectivity, reflectivity)(uint8_t, ring, ring) (uint16_t, noise, noise) (
    uint32_t, range,
    range)
)

// Use the Velodyne point format as a common representation
using PointXYZIRT = VelodynePointXYZIRT;

const int queueLength = 2000;

std::mutex imuLock;
std::mutex odoLock;

unsigned int indexNextTimeOf(const std::deque<nav_msgs::Odometry> & queue, const double time)
{
  for (unsigned int i = 0; i < queue.size(); ++i) {
    if (timeInSec(queue[i].header) < time) {
      continue;
    }
    return i;
  }
  return queue.size() - 1;
}

bool ringIsAvailable(const sensor_msgs::PointCloud2 & pointcloud)
{
  for (const auto & field : pointcloud.fields) {
    if (field.name == "ring") {
      return true;
    }
  }
  return false;
}

bool timeStampIsAvailable(const sensor_msgs::PointCloud2 & pointcloud)
{
  for (auto & field : pointcloud.fields) {
    if (field.name == "time" || field.name == "t" || field.name == "time_stamp") {
      return true;
    }
  }
  return false;
}

float rad2deg(const float rad)
{
  return rad * 180 / M_PI;
}

pcl::PointCloud<PointXYZIRT> convert(
  const sensor_msgs::PointCloud2 & cloud_msg,
  const SensorType & sensor)
{
  if (sensor == SensorType::VELODYNE) {
    return getPointCloud<PointXYZIRT>(cloud_msg);
  }

  if (sensor == SensorType::OUSTER) {
    // Convert to Velodyne format
    pcl::PointCloud<OusterPointXYZIRT> tmpOusterCloudIn =
      getPointCloud<OusterPointXYZIRT>(cloud_msg);

    pcl::PointCloud<PointXYZIRT> input_cloud;
    input_cloud.points.resize(tmpOusterCloudIn.size());
    input_cloud.is_dense = tmpOusterCloudIn.is_dense;
    for (size_t i = 0; i < tmpOusterCloudIn.size(); i++) {
      auto & src = tmpOusterCloudIn.points[i];
      auto & dst = input_cloud.points[i];
      dst.x = src.x;
      dst.y = src.y;
      dst.z = src.z;
      dst.intensity = src.intensity;
      dst.ring = src.ring;
      dst.time = src.t * 1e-9f;
    }
    return input_cloud;
  }

  throw std::runtime_error("Unknown sensor type");
}

Eigen::Vector3d interpolatePose(
  const Eigen::Vector3d & rot0, const Eigen::Vector3d & rot1,
  const double t0, const double t1, const double t)
{
  return rot1 * (t - t0) / (t1 - t0) + rot0 * (t1 - t) / (t1 - t0);
}

int findIndex(const std::vector<double> & timestamps, const double point_time)
{
  for (unsigned int i = 0; i < timestamps.size() - 1; i++) {
    if (point_time < timestamps[i]) {
      return i;
    }
  }
  return timestamps.size() - 1;
}

class RotationFinder
{
public:
  RotationFinder(
    const double scan_start_time,
    const std::vector<Eigen::Vector3d> & angles,
    const std::vector<double> & timestamps)
  : scan_start_time(scan_start_time), angles(angles), timestamps(timestamps)
  {
  }

  Eigen::Vector3d operator()(const double time_from_start) const
  {
    const double point_time = scan_start_time + time_from_start;

    const int index = findIndex(timestamps, point_time);

    if (point_time > timestamps[index] || index == 0) {
      return angles[index];
    }

    return interpolatePose(
      angles[index - 1], angles[index - 0],
      timestamps[index - 1], timestamps[index - 0],
      point_time
    );
  }

private:
  const double scan_start_time;
  const std::vector<Eigen::Vector3d> angles;
  const std::vector<double> timestamps;
};

class PositionFinder
{
public:
  PositionFinder(
    const Eigen::Vector3d & odomInc,
    const double scan_start_time,
    const double scan_end_time)
  : odomInc(odomInc),
    scan_start_time(scan_start_time),
    scan_end_time(scan_end_time) {}

  Eigen::Vector3d operator()(const double time) const
  {
    const float ratio = time / (scan_end_time - scan_start_time);
    return ratio * odomInc;
  }

private:
  const Eigen::Vector3d & odomInc;
  const double scan_start_time;
  const double scan_end_time;
};

std::tuple<Eigen::MatrixXd, Points<PointType>::type>
projectPointCloud(
  const pcl::PointCloud<PointXYZIRT> & input_points,
  const float range_min,
  const float range_max,
  const int downsampleRate,
  const int N_SCAN,
  const int Horizon_SCAN,
  const bool imuAvailable,
  const RotationFinder & calc_rotation,
  const PositionFinder & calc_position)
{
  bool firstPointFlag = true;
  Eigen::Affine3d transStartInverse;

  Eigen::MatrixXd rangeMat = -1.0 * Eigen::MatrixXd::Ones(N_SCAN, Horizon_SCAN);

  Points<PointType>::type output_points(N_SCAN * Horizon_SCAN);

  for (const PointXYZIRT & p : input_points) {
    const Eigen::Vector3d q(p.x, p.y, p.z);

    const float range = q.norm();
    if (range < range_min || range_max < range) {
      continue;
    }

    const int row_index = p.ring;

    if (row_index % downsampleRate != 0) {
      continue;
    }

    const float angle = rad2deg(atan2(q.x(), q.y()));
    const int f = static_cast<int>(Horizon_SCAN * (angle - 90.0) / 360.0);
    const int c = Horizon_SCAN / 2 - f;
    const int column_index = c % Horizon_SCAN;

    if (rangeMat(row_index, column_index) >= 0) {
      continue;
    }

    rangeMat(row_index, column_index) = range;

    const int index = column_index + row_index * Horizon_SCAN;

    if (!imuAvailable) {
      output_points[index] = makePoint(q, p.intensity);
      continue;
    }

    const Eigen::Affine3d transform = makeAffine(calc_rotation(p.time), calc_position(p.time));

    if (firstPointFlag) {
      transStartInverse = transform.inverse();
      firstPointFlag = false;
    }

    // transform points to start
    const Eigen::Affine3d transBt = transStartInverse * transform;
    output_points[index] = makePoint(transBt * q, p.intensity);
  }

  return {rangeMat, output_points};
}

bool odometryIsAvailable(
  const std::deque<nav_msgs::Odometry> & odomQueue,
  const double scan_start_time,
  const double scan_end_time)
{
  return !(
    odomQueue.empty() ||
    timeInSec(odomQueue.front().header) > scan_start_time ||
    timeInSec(odomQueue.back().header) < scan_end_time);
}

bool doOdomDeskew(
  const boost::array<double, 36> & covariance0,
  const boost::array<double, 36> & covariance1)
{
  return static_cast<int>(round(covariance0[0])) == static_cast<int>(round(covariance1[0]));
}

Eigen::Vector3d findInitialImu(
  const std::deque<sensor_msgs::Imu> & imu_buffer,
  const double scan_start_time)
{

  if (imu_buffer.empty()) {
    return Eigen::Vector3d::Zero();
  }

  Eigen::Vector3d initialIMU = Eigen::Vector3d::Zero();

  for (const sensor_msgs::Imu & imu : imu_buffer) {
    if (timeInSec(imu.header) <= scan_start_time) {
      initialIMU = quaternionToRPY(imu.orientation);
    }
  }
  return initialIMU;
}

std::tuple<std::vector<double>, std::vector<Eigen::Vector3d>> imuIncrementalOdometry(
  const double scan_end_time,
  const std::deque<sensor_msgs::Imu> & imu_buffer)
{
  std::vector<double> timestamps;
  std::vector<Eigen::Vector3d> angles;

  if (imu_buffer.empty()) {
    return {timestamps, angles};
  }

  for (const sensor_msgs::Imu & imu : imu_buffer) {
    const double time = timeInSec(imu.header);

    if (time > scan_end_time + 0.01) {
      break;
    }

    if (timestamps.size() == 0) {
      angles.push_back(Eigen::Vector3d::Zero());
      timestamps.push_back(time);
      continue;
    }

    const Eigen::Vector3d angular = vector3ToEigen(imu.angular_velocity);
    const double dt = time - timestamps.back();
    angles.push_back(angles.back() + angular * dt);
    timestamps.push_back(time);
  }
  return {timestamps, angles};
}

bool checkImuTime(
  const std::deque<sensor_msgs::Imu> & imu_buffer,
  const double scan_start_time,
  const double scan_end_time)
{

  std::lock_guard<std::mutex> lock1(imuLock);
  std::lock_guard<std::mutex> lock2(odoLock);

  // make sure IMU data available for the scan
  if (imu_buffer.empty()) {
    ROS_DEBUG("IMU queue empty ...");
    return false;
  }

  if (timeInSec(imu_buffer.front().header) > scan_start_time) {
    ROS_DEBUG("IMU time = %f", timeInSec(imu_buffer.front().header));
    ROS_DEBUG("LiDAR time = %f", scan_start_time);
    ROS_DEBUG("Timestamp of IMU data too late");
    return false;
  }

  if (timeInSec(imu_buffer.back().header) < scan_end_time) {
    ROS_DEBUG("Timestamp of IMU data too early");
    return false;
  }

  return true;
}

class ImageProjection : public ParamServer
{
private:
  const ros::Subscriber subImu;
  const ros::Subscriber subOdom;
  const ros::Subscriber subLaserCloud;

  const ros::Publisher pubExtractedCloud;
  const ros::Publisher pubLaserCloudInfo;

  std::deque<nav_msgs::Odometry> odomQueue;

  std::deque<sensor_msgs::PointCloud2> cloudQueue;

  std::deque<sensor_msgs::Imu> imu_buffer;
  const IMUConverter imu_converter_;

public:
  ImageProjection()
  : subImu(nh.subscribe(
        imuTopic, 2000,
        &ImageProjection::imuHandler, this,
        ros::TransportHints().tcpNoDelay())),
    subOdom(nh.subscribe<nav_msgs::Odometry>(
        odomTopic + "_incremental", 2000,
        &ImageProjection::odometryHandler, this,
        ros::TransportHints().tcpNoDelay())),
    subLaserCloud(nh.subscribe<sensor_msgs::PointCloud2>(
        pointCloudTopic, 5,
        &ImageProjection::cloudHandler, this, ros::TransportHints().tcpNoDelay())),
    pubExtractedCloud(
      nh.advertise<sensor_msgs::PointCloud2>("lio_sam/deskew/cloud_deskewed", 1)),
    pubLaserCloudInfo(
      nh.advertise<lio_sam::cloud_info>("lio_sam/deskew/cloud_info", 1))
  {
    pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
  }

  ~ImageProjection() {}

  void imuHandler(const sensor_msgs::Imu::ConstPtr & imuMsg)
  {
    const sensor_msgs::Imu msg = imu_converter_.imuConverter(*imuMsg);

    std::lock_guard<std::mutex> lock1(imuLock);
    imu_buffer.push_back(msg);
  }

  void odometryHandler(const nav_msgs::Odometry::ConstPtr & odometryMsg)
  {
    std::lock_guard<std::mutex> lock2(odoLock);
    odomQueue.push_back(*odometryMsg);
  }

  void cloudHandler(const sensor_msgs::PointCloud2ConstPtr & laserCloudMsg)
  {
    cloudQueue.push_back(*laserCloudMsg);
    if (cloudQueue.size() <= 2) {
      return;
    }

    const sensor_msgs::PointCloud2 cloud_msg = cloudQueue.front();
    cloudQueue.pop_front();

    const pcl::PointCloud<PointXYZIRT> input_cloud = [&] {
        try {
          return convert(cloud_msg, sensor);
        } catch (const std::runtime_error & e) {
          ROS_ERROR_STREAM("Unknown sensor type: " << int(sensor));
          ros::shutdown();
          return pcl::PointCloud<PointXYZIRT>();
        }
      } ();

    if (!input_cloud.is_dense) {
      ROS_ERROR("Point cloud is not in dense format, please remove NaN points first!");
      ros::shutdown();
    }

    if (!ringIsAvailable(cloud_msg)) {
      ROS_ERROR("Point cloud ring channel could not be found");
      ros::shutdown();
    }

    if (!timeStampIsAvailable(cloud_msg)) {
      ROS_ERROR("Point cloud timestamp not available");
      ros::shutdown();
    }

    const double scan_start_time = timeInSec(cloud_msg.header);
    const double scan_end_time = scan_start_time + input_cloud.back().time;

    if (!checkImuTime(imu_buffer, scan_start_time, scan_end_time)) {
      return;
    }

    lio_sam::cloud_info cloudInfo;
    cloudInfo.startRingIndex.assign(N_SCAN, 0);
    cloudInfo.endRingIndex.assign(N_SCAN, 0);

    cloudInfo.pointColInd.assign(N_SCAN * Horizon_SCAN, 0);
    cloudInfo.pointRange.assign(N_SCAN * Horizon_SCAN, 0);

    {
      std::lock_guard<std::mutex> lock1(imuLock);
      dropBefore(scan_start_time - 0.01, imu_buffer);
    }

    cloudInfo.initialIMU = eigenToVector3(findInitialImu(imu_buffer, scan_start_time));

    const auto [timestamps, angles] = imuIncrementalOdometry(scan_end_time, imu_buffer);

    {
      std::lock_guard<std::mutex> lock2(odoLock);
      dropBefore(scan_start_time - 0.01, odomQueue);
    }

    const bool odomAvailable = odometryIsAvailable(odomQueue, scan_start_time, scan_end_time);

    Eigen::Vector3d odomInc = Eigen::Vector3d::Zero();

    if (odomAvailable) {
      const unsigned int index0 = indexNextTimeOf(odomQueue, scan_start_time);
      const nav_msgs::Odometry msg0 = odomQueue[index0];

      const unsigned int index1 = indexNextTimeOf(odomQueue, scan_end_time);
      const nav_msgs::Odometry msg1 = odomQueue[index1];

      cloudInfo.initial_pose = msg0.pose.pose;

      if (doOdomDeskew(msg0.pose.covariance, msg1.pose.covariance)) {
        const Eigen::Affine3d p0 = poseToAffine(msg0.pose.pose);
        const Eigen::Affine3d p1 = poseToAffine(msg1.pose.pose);
        odomInc = (p0.inverse() * p1).translation();
      }
    }

    cloudInfo.odomAvailable = odomAvailable;
    cloudInfo.imuAvailable = timestamps.size() > 1;

    const RotationFinder calc_rotation(scan_start_time, angles, timestamps);
    const PositionFinder calc_position(odomInc, scan_start_time, scan_end_time);

    const auto [rangeMat, output_points] = projectPointCloud(
      input_cloud, range_min, range_max,
      downsampleRate, N_SCAN, Horizon_SCAN,
      cloudInfo.imuAvailable, calc_rotation, calc_position
    );

    pcl::PointCloud<PointType> extractedCloud;
    int count = 0;
    // extract segmented cloud for lidar odometry
    for (int i = 0; i < N_SCAN; ++i) {
      cloudInfo.startRingIndex[i] = count + 5;

      for (int j = 0; j < Horizon_SCAN; ++j) {
        const float range = rangeMat(i, j);
        if (range < 0) {
          continue;
        }

        // mark the points' column index for marking occlusion later
        cloudInfo.pointColInd[count] = j;
        // save range info
        cloudInfo.pointRange[count] = range;
        // save extracted cloud
        extractedCloud.push_back(output_points[j + i * Horizon_SCAN]);
        // size of extracted cloud
        count += 1;
      }

      cloudInfo.endRingIndex[i] = count - 5;
    }

    cloudInfo.header = cloud_msg.header;
    cloudInfo.cloud_deskewed = publishCloud(
      pubExtractedCloud, extractedCloud,
      cloud_msg.header.stamp, lidarFrame);
    pubLaserCloudInfo.publish(cloudInfo);
  }
};

int main(int argc, char ** argv)
{
  ros::init(argc, argv, "lio_sam");

  ImageProjection IP;

  ROS_INFO("\033[1;32m----> Image Projection Started.\033[0m");

  ros::MultiThreadedSpinner spinner(3);
  spinner.spin();

  return 0;
}
