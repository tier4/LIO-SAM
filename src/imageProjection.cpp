#include "message.hpp"
#include "utility.hpp"
#include "param_server.h"
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

unsigned int indexNextTimeOf(
  const std::deque<geometry_msgs::TransformStamped> & queue,
  const double time)
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
    return *getPointCloud<PointXYZIRT>(cloud_msg);
  }

  if (sensor == SensorType::OUSTER) {
    // Convert to Velodyne format
    const auto cloud = getPointCloud<OusterPointXYZIRT>(cloud_msg);

    pcl::PointCloud<PointXYZIRT> input_cloud;
    input_cloud.points.resize(cloud->size());
    input_cloud.is_dense = cloud->is_dense;
    for (size_t i = 0; i < cloud->size(); i++) {
      auto & src = cloud->points[i];
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

int findIndex(const std::vector<double> & imu_timestamps, const double point_time)
{
  for (unsigned int i = 0; i < imu_timestamps.size() - 1; i++) {
    if (point_time < imu_timestamps[i]) {
      return i;
    }
  }
  return imu_timestamps.size() - 1;
}

Eigen::Vector3d calcRotation(
  const double scan_start_time,
  const std::vector<Eigen::Vector3d> & angles,
  const std::vector<double> & imu_timestamps,
  const double time_from_start)
{
  const double point_time = scan_start_time + time_from_start;

  const int index = findIndex(imu_timestamps, point_time);

  if (index == 0 || index == static_cast<int>(imu_timestamps.size()) - 1) {
    return angles[index];
  }

  return interpolatePose(
    angles[index - 1], angles[index - 0],
    imu_timestamps[index - 1], imu_timestamps[index - 0],
    point_time
  );
}

Eigen::Vector3d calcPosition(
  const Eigen::Vector3d & imu_incremental_odometry,
  const double scan_start_time, const double scan_end_time, const double time)
{
  return imu_incremental_odometry * time / (scan_end_time - scan_start_time);
}

int calcColumnIndex(const int Horizon_SCAN, const double x, const double y)
{
  const float angle = rad2deg(atan2(x, y));
  const int f = static_cast<int>(Horizon_SCAN * (angle - 90.0) / 360.0);
  const int c = Horizon_SCAN / 2 - f;
  const int column_index = c % Horizon_SCAN;
  return column_index;
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

class PointCloudProjection
{
public:
  PointCloudProjection(
    const float range_min, const float range_max,
    const int downsampleRate, const int N_SCAN, const int Horizon_SCAN)
  : range_min(range_min), range_max(range_max),
    downsampleRate(downsampleRate), N_SCAN(N_SCAN), Horizon_SCAN(Horizon_SCAN)
  {
  }

  std::tuple<Eigen::MatrixXd, std::vector<PointType>, bool>
  compute(
    const std::deque<sensor_msgs::Imu> & imu_buffer,
    const pcl::PointCloud<PointXYZIRT> & input_points,
    const double scan_start_time,
    const double scan_end_time,
    const Eigen::Vector3d & imu_incremental_odometry) const
  {
    const auto [imu_timestamps, angles] = imuIncrementalOdometry(scan_end_time, imu_buffer);

    const bool imu_available = imu_timestamps.size() > 1;

    bool is_first_point = true;
    Eigen::Affine3d start_inverse;

    Eigen::MatrixXd range_matrix = -1.0 * Eigen::MatrixXd::Ones(N_SCAN, Horizon_SCAN);

    std::vector<PointType> output_points(N_SCAN * Horizon_SCAN);

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

      const int column_index = calcColumnIndex(Horizon_SCAN, q.x(), q.y());

      if (range_matrix(row_index, column_index) >= 0) {
        continue;
      }

      range_matrix(row_index, column_index) = range;

      const int index = column_index + row_index * Horizon_SCAN;

      if (!imu_available) {
        output_points[index] = makePoint(q, p.intensity);
        continue;
      }

      const Eigen::Affine3d transform = makeAffine(
        calcRotation(scan_start_time, angles, imu_timestamps, p.time),
        calcPosition(imu_incremental_odometry, scan_start_time, scan_end_time, p.time)
      );

      if (is_first_point) {
        start_inverse = transform.inverse();
        is_first_point = false;
      }

      // transform points to start
      output_points[index] = makePoint((start_inverse * transform) * q, p.intensity);
    }

    return {range_matrix, output_points, imu_available};
  }

private:
  const float range_min;
  const float range_max;
  const int downsampleRate;
  const int N_SCAN;
  const int Horizon_SCAN;
};

bool imuOdometryAvailable(
  const std::deque<geometry_msgs::TransformStamped> & odomQueue,
  const double scan_start_time,
  const double scan_end_time)
{
  return !(
    odomQueue.empty() ||
    timeInSec(odomQueue.front().header) > scan_start_time ||
    timeInSec(odomQueue.back().header) < scan_end_time);
}

Eigen::Vector3d findImuOrientation(
  const std::deque<sensor_msgs::Imu> & imu_buffer,
  const double scan_start_time)
{
  if (imu_buffer.empty()) {
    return Eigen::Vector3d::Zero();
  }

  Eigen::Vector3d imu_orientation = Eigen::Vector3d::Zero();

  for (const sensor_msgs::Imu & imu : imu_buffer) {
    if (timeInSec(imu.header) <= scan_start_time) {
      imu_orientation = quaternionToRPY(imu.orientation);
    }
  }
  return imu_orientation;
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

geometry_msgs::TransformStamped odomNextOf(
  const std::deque<geometry_msgs::TransformStamped> & odomQueue,
  const double time)
{
  const unsigned int index = indexNextTimeOf(odomQueue, time);
  return odomQueue[index];
}

class ImageProjection : public ParamServer
{
private:
  const ros::Subscriber subImu;
  const ros::Subscriber subOdom;
  const ros::Subscriber subLaserCloud;

  const ros::Publisher pubExtractedCloud;
  const ros::Publisher pubLaserCloudInfo;
  const PointCloudProjection projection_;

  std::deque<geometry_msgs::TransformStamped> odomQueue;

  std::deque<sensor_msgs::PointCloud2> cloudQueue;

  std::deque<sensor_msgs::Imu> imu_buffer;
  const IMUConverter imu_converter_;

public:
  ImageProjection()
  : subImu(nh.subscribe(
        imuTopic, 2000,
        &ImageProjection::imuHandler, this,
        ros::TransportHints().tcpNoDelay())),
    subOdom(nh.subscribe<geometry_msgs::TransformStamped>(
        odomTopic + "_incremental", 2000,
        &ImageProjection::odometryHandler, this,
        ros::TransportHints().tcpNoDelay())),
    subLaserCloud(nh.subscribe<sensor_msgs::PointCloud2>(
        pointCloudTopic, 5,
        &ImageProjection::cloudHandler, this, ros::TransportHints().tcpNoDelay())),
    pubExtractedCloud(
      nh.advertise<sensor_msgs::PointCloud2>("lio_sam/deskew/cloud_deskewed", 1)),
    pubLaserCloudInfo(
      nh.advertise<lio_sam::cloud_info>("lio_sam/deskew/cloud_info", 1)),
    projection_(PointCloudProjection(range_min, range_max, downsampleRate, N_SCAN, Horizon_SCAN))
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

  void odometryHandler(const geometry_msgs::TransformStamped::ConstPtr & odometryMsg)
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
    cloudInfo.ring_start_indices.assign(N_SCAN, 0);
    cloudInfo.end_ring_indices.assign(N_SCAN, 0);

    cloudInfo.pointColInd.assign(N_SCAN * Horizon_SCAN, 0);
    cloudInfo.pointRange.assign(N_SCAN * Horizon_SCAN, 0);

    {
      std::lock_guard<std::mutex> lock1(imuLock);
      dropBefore(scan_start_time - 0.01, imu_buffer);
    }

    cloudInfo.imu_orientation = eigenToVector3(findImuOrientation(imu_buffer, scan_start_time));

    {
      std::lock_guard<std::mutex> lock2(odoLock);
      dropBefore(scan_start_time - 0.01, odomQueue);
    }

    const bool imu_odometry_available = imuOdometryAvailable(
      odomQueue, scan_start_time, scan_end_time
    );

    Eigen::Vector3d imu_incremental_odometry = Eigen::Vector3d::Zero();

    if (imu_odometry_available) {
      const geometry_msgs::TransformStamped msg0 = odomNextOf(odomQueue, scan_start_time);
      const geometry_msgs::TransformStamped msg1 = odomNextOf(odomQueue, scan_end_time);

      cloudInfo.scan_start_imu_pose = transformToPose(msg0.transform);

      const Eigen::Affine3d p0 = transformToAffine(msg0.transform);
      const Eigen::Affine3d p1 = transformToAffine(msg1.transform);
      imu_incremental_odometry = (p0.inverse() * p1).translation();
    }

    const auto [range_matrix, output_points, imu_available] = projection_.compute(
      imu_buffer, input_cloud, scan_start_time, scan_end_time, imu_incremental_odometry);

    cloudInfo.imu_odometry_available = imu_odometry_available;
    cloudInfo.imu_orientation_available = imu_available;

    pcl::PointCloud<PointType> extractedCloud;
    int count = 0;
    // extract segmented cloud for lidar odometry
    for (int i = 0; i < N_SCAN; ++i) {
      cloudInfo.ring_start_indices[i] = count + 5;

      for (int j = 0; j < Horizon_SCAN; ++j) {
        const float range = range_matrix(i, j);
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

      cloudInfo.end_ring_indices[i] = count - 5;
    }

    cloudInfo.header = cloud_msg.header;
    cloudInfo.cloud_deskewed = toRosMsg(extractedCloud, cloud_msg.header.stamp, lidarFrame);
    pubExtractedCloud.publish(cloudInfo.cloud_deskewed);
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
