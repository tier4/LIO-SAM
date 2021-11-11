#include "message.hpp"
#include "utility.hpp"
#include "param_server.h"
#include "lio_sam/cloud_info.h"

#include "range/v3/all.hpp"

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

// TODO use binary search
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

Eigen::Vector3d interpolate3d(
  const Eigen::Vector3d & v0, const Eigen::Vector3d & v1,
  const double t0, const double t1, const double t)
{
  return v1 * (t - t0) / (t1 - t0) + v0 * (t1 - t) / (t1 - t0);
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
  const std::vector<Eigen::Vector3d> & angles,
  const std::vector<double> & timestamps,
  const double time)
{
  const int i = findIndex(timestamps, time);

  if (i == 0 || i == static_cast<int>(timestamps.size()) - 1) {
    return angles[i];
  }

  return interpolate3d(angles[i - 1], angles[i], timestamps[i - 1], timestamps[i], time);
}

Eigen::Vector3d calcPosition(
  const Eigen::Vector3d & translation_within_scan,
  const double scan_start_time, const double scan_end_time, const double time)
{
  const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
  const double t = scan_start_time + time;
  return interpolate3d(zero, translation_within_scan, scan_start_time, scan_end_time, t);
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

Eigen::MatrixXd makeRangeMatrix(
  const pcl::PointCloud<PointXYZIRT> & input_points,
  const float range_min, const float range_max,
  const int N_SCAN, const int Horizon_SCAN)
{
  const auto f = [&](const PointXYZIRT & p) {
      const Eigen::Vector3d q(p.x, p.y, p.z);
      const int row_index = p.ring;
      const int column_index = calcColumnIndex(Horizon_SCAN, q.x(), q.y());
      return std::make_tuple(q.norm(), row_index, column_index);
    };

  const auto iterator = input_points | ranges::views::transform(f);

  Eigen::MatrixXd range_matrix = -1.0 * Eigen::MatrixXd::Ones(N_SCAN, Horizon_SCAN);
  for (const auto & [range, row_index, column_index] : iterator) {
    if (range < range_min || range_max < range) {
      continue;
    }

    if (range_matrix(row_index, column_index) >= 0) {
      continue;
    }

    range_matrix(row_index, column_index) = range;
  }

  return range_matrix;
}

std::vector<pcl::PointXYZ> projectWithoutImu(
  const pcl::PointCloud<PointXYZIRT> & input_points,
  const Eigen::MatrixXd & range_matrix,
  const int N_SCAN, const int Horizon_SCAN)
{
  const auto f = [&](const PointXYZIRT & p) {
      const Eigen::Vector3d q(p.x, p.y, p.z);

      const int row_index = p.ring;
      const int column_index = calcColumnIndex(Horizon_SCAN, q.x(), q.y());

      return std::make_tuple(makePointXYZ(q), row_index, column_index);
    };
  const auto iterator = input_points | ranges::views::transform(f);
  std::vector<pcl::PointXYZ> output_points(N_SCAN * Horizon_SCAN);
  for (const auto & [q, row_index, column_index] : iterator) {
    if (range_matrix(row_index, column_index) < 0) {
      continue;
    }

    const int index = column_index + row_index * Horizon_SCAN;
    output_points[index] = q;
  }
  return output_points;
}

std::vector<pcl::PointXYZ> projectWithImu(
  const pcl::PointCloud<PointXYZIRT> & input_points,
  const Eigen::MatrixXd & range_matrix,
  const std::vector<double> & imu_timestamps,
  const std::vector<Eigen::Vector3d> & angles,
  const int N_SCAN, const int Horizon_SCAN,
  const double scan_start_time,
  const double scan_end_time,
  const Eigen::Vector3d & translation_within_scan)
{
  const auto f = [&](const PointXYZIRT & p) {
      const Eigen::Vector3d q(p.x, p.y, p.z);

      const int row_index = p.ring;
      const int column_index = calcColumnIndex(Horizon_SCAN, q.x(), q.y());

      return std::make_tuple(q, p.time, row_index, column_index);
    };

  const auto iterator = input_points | ranges::views::transform(f);

  bool is_first_point = true;
  Eigen::Affine3d start_inverse;
  std::vector<pcl::PointXYZ> output_points(N_SCAN * Horizon_SCAN);

  for (const auto & [q, time, row_index, column_index] : iterator) {
    if (range_matrix(row_index, column_index) < 0) {
      continue;
    }

    const Eigen::Affine3d transform = makeAffine(
      calcRotation(angles, imu_timestamps, scan_start_time + time),
      calcPosition(translation_within_scan, scan_start_time, scan_end_time, time)
    );

    if (is_first_point) {
      start_inverse = transform.inverse();
      is_first_point = false;
    }

    const int index = column_index + row_index * Horizon_SCAN;
    output_points[index] = makePointXYZ((start_inverse * transform) * q);
  }

  return output_points;
}

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

bool scanTimesAreWithinImu(
  const std::deque<sensor_msgs::Imu> & imu_buffer,
  const double scan_start_time,
  const double scan_end_time)
{
  return !imu_buffer.empty() &&
         timeInSec(imu_buffer.front().header) <= scan_start_time &&
         timeInSec(imu_buffer.back().header) >= scan_end_time;
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

  std::deque<geometry_msgs::TransformStamped> imu_odometry_queue_;

  std::deque<sensor_msgs::PointCloud2> cloudQueue;

  std::deque<sensor_msgs::Imu> imu_buffer;
  const IMUExtrinsic imu_extrinsic_;

public:
  ImageProjection()
  : subImu(nh.subscribe(
        imuTopic, 2000,
        &ImageProjection::imuHandler, this,
        ros::TransportHints().tcpNoDelay())),
    subOdom(nh.subscribe<geometry_msgs::TransformStamped>(
        imu_incremental_odometry_topic,
        2000, &ImageProjection::imuOdometryHandler, this,
        ros::TransportHints().tcpNoDelay())),
    subLaserCloud(nh.subscribe<sensor_msgs::PointCloud2>(
        pointCloudTopic, 5,
        &ImageProjection::cloudHandler, this, ros::TransportHints().tcpNoDelay())),
    pubExtractedCloud(
      nh.advertise<sensor_msgs::PointCloud2>("lio_sam/deskew/cloud_deskewed", 1)),
    pubLaserCloudInfo(
      nh.advertise<lio_sam::cloud_info>("lio_sam/deskew/cloud_info", 1)),
    imu_extrinsic_(IMUExtrinsic(extRot, extQRPY))
  {
    pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
  }

  ~ImageProjection() {}

  void imuHandler(const sensor_msgs::Imu::ConstPtr & imuMsg)
  {
    const sensor_msgs::Imu msg = imu_extrinsic_.transform(*imuMsg);

    std::lock_guard<std::mutex> lock1(imuLock);
    imu_buffer.push_back(msg);
  }

  void imuOdometryHandler(const geometry_msgs::TransformStamped::ConstPtr & odometryMsg)
  {
    std::lock_guard<std::mutex> lock2(odoLock);
    imu_odometry_queue_.push_back(*odometryMsg);
  }

  pcl::PointCloud<PointXYZIRT> msgToPointCloud(
    const sensor_msgs::PointCloud2 cloud_msg,
    const SensorType & sensor)
  {
    try {
      return convert(cloud_msg, sensor);
    } catch (const std::runtime_error & e) {
      ROS_ERROR_STREAM("Unknown sensor type: " << int(sensor));
      ros::shutdown();
      return pcl::PointCloud<PointXYZIRT>();
    }
  }

  void cloudHandler(const sensor_msgs::PointCloud2ConstPtr & laserCloudMsg)
  {
    cloudQueue.push_back(*laserCloudMsg);
    if (cloudQueue.size() <= 2) {
      return;
    }

    const sensor_msgs::PointCloud2 cloud_msg = cloudQueue.front();
    cloudQueue.pop_front();

    const pcl::PointCloud<PointXYZIRT> input_points = msgToPointCloud(cloud_msg, sensor);

    if (!input_points.is_dense) {
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
    const double scan_end_time = scan_start_time + input_points.back().time;

    if (!scanTimesAreWithinImu(imu_buffer, scan_start_time, scan_end_time)) {
      return;
    }

    lio_sam::cloud_info cloud_info;
    {
      std::lock_guard<std::mutex> lock1(imuLock);
      dropBefore(scan_start_time - 0.01, imu_buffer);
    }

    cloud_info.imu_orientation = eigenToVector3(findImuOrientation(imu_buffer, scan_start_time));

    {
      std::lock_guard<std::mutex> lock2(odoLock);
      dropBefore(scan_start_time - 0.01, imu_odometry_queue_);
    }

    const bool imu_odometry_available = imuOdometryAvailable(
      imu_odometry_queue_, scan_start_time, scan_end_time
    );

    Eigen::Vector3d translation_within_scan = Eigen::Vector3d::Zero();

    if (imu_odometry_available) {
      const auto msg0 = odomNextOf(imu_odometry_queue_, scan_start_time);
      const auto msg1 = odomNextOf(imu_odometry_queue_, scan_end_time);

      cloud_info.scan_start_imu_pose = transformToPose(msg0.transform);

      const Eigen::Affine3d p0 = transformToAffine(msg0.transform);
      const Eigen::Affine3d p1 = transformToAffine(msg1.transform);
      translation_within_scan = (p0.inverse() * p1).translation();
    }
    cloud_info.imu_odometry_available = imu_odometry_available;

    const auto [imu_timestamps, angles] = imuIncrementalOdometry(scan_end_time, imu_buffer);
    const bool imu_available = imu_timestamps.size() > 1;

    const Eigen::MatrixXd range_matrix = makeRangeMatrix(
      input_points, range_min, range_max, N_SCAN, Horizon_SCAN);

    std::vector<pcl::PointXYZ> output_points;
    if (imu_available) {
      output_points = projectWithImu(
        input_points, range_matrix,
        imu_timestamps, angles,
        N_SCAN, Horizon_SCAN,
        scan_start_time, scan_end_time,
        translation_within_scan);
    } else {
      output_points = projectWithoutImu(input_points, range_matrix, N_SCAN, Horizon_SCAN);
    }

    cloud_info.imu_orientation_available = imu_available;

    cloud_info.ring_start_indices.assign(N_SCAN, 0);
    cloud_info.end_ring_indices.assign(N_SCAN, 0);

    cloud_info.point_column_indices.assign(N_SCAN * Horizon_SCAN, 0);
    cloud_info.point_range.assign(N_SCAN * Horizon_SCAN, 0);

    pcl::PointCloud<pcl::PointXYZ> cloud;
    int count = 0;
    // extract segmented cloud for lidar odometry
    for (int i = 0; i < N_SCAN; ++i) {
      cloud_info.ring_start_indices[i] = count + 5;

      for (int j = 0; j < Horizon_SCAN; ++j) {
        const float range = range_matrix(i, j);
        if (range < 0) {
          continue;
        }

        // mark the points' column index for marking occlusion later
        cloud_info.point_column_indices[count] = j;
        // save range info
        cloud_info.point_range[count] = range;
        // save extracted cloud
        cloud.push_back(output_points[j + i * Horizon_SCAN]);
        // size of extracted cloud
        count += 1;
      }

      cloud_info.end_ring_indices[i] = count - 5;
    }

    cloud_info.header = cloud_msg.header;
    cloud_info.cloud_deskewed = toRosMsg(cloud, cloud_msg.header.stamp, lidarFrame);
    pubExtractedCloud.publish(cloud_info.cloud_deskewed);
    pubLaserCloudInfo.publish(cloud_info);
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
