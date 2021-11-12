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

int calcColumnIndex(const int Horizon_SCAN, const double x, const double y)
{
  const double angle = rad2deg(atan2(y, x));  // [-180 ~ 180]
  const double k = Horizon_SCAN * angle / (180.0 * 2.0);  // [-Horizon_SCAN / 2 ~ Horizon_SCAN / 2]
  const double u = k + Horizon_SCAN / 2.0;
  return static_cast<int>(u);
}

std::tuple<std::vector<double>, std::vector<Eigen::Quaterniond>> imuIncrementalOdometry(
  const double scan_start_time,
  const double scan_end_time,
  const std::deque<sensor_msgs::Imu> & imu_buffer)
{
  std::vector<double> timestamps;
  std::vector<Eigen::Quaterniond> quaternions;  // TODO replace with quaternion

  if (imu_buffer.empty()) {
    return {timestamps, quaternions};
  }

  for (const sensor_msgs::Imu & imu : imu_buffer) {
    const double time = timeInSec(imu.header) - scan_start_time;

    if (time > scan_end_time - scan_start_time + 0.01) {
      break;
    }

    if (timestamps.size() == 0) {
      quaternions.push_back(Eigen::Quaterniond::Identity());
      timestamps.push_back(time);
      continue;
    }

    const Eigen::Vector3d angular = vector3ToEigen(imu.angular_velocity);
    const double dt = time - timestamps.back();
    const Eigen::Vector3d w = angular * dt;
    Eigen::Quaterniond dq(Eigen::AngleAxis(w.norm(), w.normalized()));
    quaternions.push_back(quaternions.back() * dq);
    timestamps.push_back(time);
  }
  return {timestamps, quaternions};
}

std::tuple<std::vector<int>, std::vector<double>, std::vector<Eigen::Vector3d>>
extranctElements(
  const pcl::PointCloud<PointXYZIRT> & input_points,
  const float range_min, const float range_max,
  const int Horizon_SCAN)
{
  const auto f = [&](const PointXYZIRT & p) {
      const int row_index = p.ring;
      const int column_index = calcColumnIndex(Horizon_SCAN, p.x, p.y);
      const int index = column_index + row_index * Horizon_SCAN;
      const Eigen::Vector3d q(p.x, p.y, p.z);
      return std::make_tuple(index, p.time, q);
    };

  std::set<int> unique_indices;
  std::vector<int> indices;
  std::vector<double> times;
  std::vector<Eigen::Vector3d> points;

  const auto iterator = input_points | ranges::views::transform(f);
  for (const auto & [index, time, point] : iterator) {
    const double range = point.norm();
    if (range < range_min || range_max < range) {
      continue;
    }

    if (unique_indices.find(index) != unique_indices.end()) {
      continue;
    }

    unique_indices.insert(index);
    indices.push_back(index);
    times.push_back(time);
    points.push_back(point);
  }

  return {indices, times, points};
}

std::unordered_map<int, double> makeRangeMatrix(
  const pcl::PointCloud<PointXYZIRT> & input_points,
  const float range_min, const float range_max,
  const int Horizon_SCAN)
{
  const auto f = [&](const PointXYZIRT & p) {
      const Eigen::Vector3d q(p.x, p.y, p.z);
      const int row_index = p.ring;
      const int column_index = calcColumnIndex(Horizon_SCAN, p.x, p.y);
      const int index = column_index + row_index * Horizon_SCAN;
      return std::make_tuple(index, q.norm());
    };

  const auto iterator = input_points | ranges::views::transform(f);

  std::unordered_map<int, double> range_map;
  for (const auto & [index, range] : iterator) {
    if (range < range_min || range_max < range) {
      continue;
    }

    if (range_map.find(index) == range_map.end()) {
      range_map[index] = range;
    }
  }

  return range_map;
}

std::unordered_map<int, pcl::PointXYZ> projectWithoutImu(
  const pcl::PointCloud<PointXYZIRT> & input_points,
  const std::unordered_map<int, double> & range_map,
  const int N_SCAN, const int Horizon_SCAN)
{
  const auto f = [&](const PointXYZIRT & p) {
      const Eigen::Vector3d q(p.x, p.y, p.z);

      const int row_index = p.ring;
      const int column_index = calcColumnIndex(Horizon_SCAN, q.x(), q.y());

      const int index = column_index + row_index * Horizon_SCAN;
      return std::make_tuple(index, makePointXYZ(q));
    };

  const auto iterator = input_points | ranges::views::transform(f);
  std::unordered_map<int, pcl::PointXYZ> output_points;
  for (const auto & [index, q] : iterator) {
    if (range_map.find(index) == range_map.end()) {
      continue;
    }
    output_points[index] = q;
  }
  return output_points;
}

std::unordered_map<int, pcl::PointXYZ> projectWithImu(
  const pcl::PointCloud<PointXYZIRT> & input_points,
  const std::unordered_map<int, double> & range_map,
  const std::vector<double> & timestamps,
  const std::vector<Eigen::Quaterniond> & quaternions,
  const int Horizon_SCAN,
  const double translation_interval,
  const Eigen::Vector3d & translation_within_scan)
{
  const auto translation = [&](const double time) {
      const Eigen::Vector3d p = translation_within_scan;
      const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
      return interpolate3d(zero, p, 0., translation_interval, time);
    };

  const auto rotation = [&](const double t) {
      const int i = findIndex(timestamps, t);
      if (i == 0 || i == static_cast<int>(timestamps.size()) - 1) {
        return quaternions[i];
      }
      return interpolate(quaternions[i - 1], quaternions[i], timestamps[i - 1], timestamps[i], t);
    };

  const auto f = [&](const PointXYZIRT & p) {
      const Eigen::Vector3d q(p.x, p.y, p.z);

      const int row_index = p.ring;
      const int column_index = calcColumnIndex(Horizon_SCAN, q.x(), q.y());

      const int index = column_index + row_index * Horizon_SCAN;
      return std::make_tuple(index, q, p.time);
    };

  const auto iterator = input_points | ranges::views::transform(f);

  std::optional<Eigen::Affine3d> start_inverse = std::nullopt;
  std::unordered_map<int, pcl::PointXYZ> output_points;
  for (const auto & [index, q, time] : iterator) {
    if (range_map.find(index) == range_map.end()) {
      continue;
    }

    const Eigen::Affine3d transform = makeAffine(rotation(time), translation(time));

    if (!start_inverse.has_value()) {
      start_inverse = transform.inverse();
    }

    output_points[index] = makePointXYZ((start_inverse.value() * transform) * q);
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
    const SensorType & sensor) const
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
    double translation_interval = 0.0;
    if (imu_odometry_available) {
      const auto msg0 = odomNextOf(imu_odometry_queue_, scan_start_time);
      const auto msg1 = odomNextOf(imu_odometry_queue_, scan_end_time);

      cloud_info.scan_start_imu_pose = transformToPose(msg0.transform);

      const Eigen::Affine3d p0 = transformToAffine(msg0.transform);
      const Eigen::Affine3d p1 = transformToAffine(msg1.transform);
      translation_within_scan = (p0.inverse() * p1).translation();
      translation_interval = scan_end_time - scan_start_time;
    }
    cloud_info.imu_odometry_available = imu_odometry_available;

    const auto [imu_timestamps, quaternions] = imuIncrementalOdometry(
      scan_start_time, scan_end_time, imu_buffer
    );
    const bool imu_available = imu_timestamps.size() > 1;

    const auto [indices, times, points] = extranctElements(
      input_points, range_min, range_max, Horizon_SCAN
    );
    const auto range_map = makeRangeMatrix(input_points, range_min, range_max, Horizon_SCAN);

    std::unordered_map<int, pcl::PointXYZ> output_points;
    if (imu_available && imu_odometry_available) {
      output_points = projectWithImu(
        input_points, range_map,
        imu_timestamps, quaternions,
        Horizon_SCAN, translation_interval, translation_within_scan);
    } else {
      output_points = projectWithoutImu(input_points, range_map, N_SCAN, Horizon_SCAN);
    }

    cloud_info.imu_orientation_available = imu_available;

    cloud_info.ring_start_indices.assign(N_SCAN, 0);
    cloud_info.end_ring_indices.assign(N_SCAN, 0);

    cloud_info.point_column_indices.assign(N_SCAN * Horizon_SCAN, 0);
    cloud_info.point_range.assign(N_SCAN * Horizon_SCAN, 0);

    pcl::PointCloud<pcl::PointXYZ> cloud;
    int count = 0;
    for (int row_index = 0; row_index < N_SCAN; ++row_index) {
      cloud_info.ring_start_indices[row_index] = count + 5;

      for (int column_index = 0; column_index < Horizon_SCAN; ++column_index) {
        const int index = column_index + row_index * Horizon_SCAN;
        if (range_map.find(index) == range_map.end()) {
          continue;
        }

        cloud_info.point_column_indices[count] = column_index;
        cloud_info.point_range[count] = range_map.at(index);
        cloud.push_back(output_points[index]);
        count += 1;
      }

      cloud_info.end_ring_indices[row_index] = count - 5;
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
