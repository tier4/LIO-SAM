#include "utility.h"
#include "lio_sam/cloud_info.h"

struct VelodynePointXYZIRT
{
  PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
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

class RotationFinder
{
public:
  RotationFinder(
    const double scan_start_time,
    const std::vector<Eigen::Vector3d> & imuRot,
    const std::vector<double> & imuTime)
  : scan_start_time(scan_start_time), imuRot(imuRot), imuTime(imuTime) {}

  Eigen::Vector3d operator()(const double relTime) const
  {
    const double point_time = scan_start_time + relTime;

    int index = imuTime.size() - 1;
    for (int i = 0; i < imuTime.size() - 1; i++) {
      if (imuTime[i] > point_time) {
        index = i;
        break;
      }
    }

    if (point_time > imuTime[index] || index == 0) {
      return imuRot[index];
    }

    return interpolatePose(
      imuRot[index - 1], imuRot[index - 0],
      imuTime[index - 1], imuTime[index - 0],
      point_time
    );
  }

private:
  const double scan_start_time;
  const std::vector<Eigen::Vector3d> imuRot;
  const std::vector<double> imuTime;
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

  Eigen::Vector3d operator()(const double relTime) const
  {
    const float ratio = relTime / (scan_end_time - scan_start_time);
    return ratio * odomInc;
  }

private:
  const Eigen::Vector3d & odomInc;
  const double scan_start_time;
  const double scan_end_time;
};

class AffineFinder
{
public:
  AffineFinder(
    const std::vector<Eigen::Vector3d> & imuRot,
    const std::vector<double> & imuTime,
    const Eigen::Vector3d & odomInc,
    const double scan_start_time,
    const double scan_end_time)
  : calc_rotation(RotationFinder(scan_start_time, imuRot, imuTime)),
    calc_position(
      PositionFinder(odomInc, scan_start_time, scan_end_time))
  {
  }

  Eigen::Affine3d operator()(const double rel_time) const
  {
    const Eigen::Vector3d r = calc_rotation(rel_time);
    const Eigen::Vector3d p = calc_position(rel_time);
    return makeAffine(r, p);
  }

private:
  const RotationFinder calc_rotation;
  const PositionFinder calc_position;
};

std::tuple<cv::Mat, Points<PointType>::type>
projectPointCloud(
  const Points<PointXYZIRT>::type & input_points,
  const float range_min,
  const float range_max,
  const int downsampleRate,
  const int N_SCAN,
  const int Horizon_SCAN,
  const bool imuAvailable,
  const AffineFinder & calc_transform)
{
  bool firstPointFlag = true;
  Eigen::Affine3d transStartInverse;

  cv::Mat rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));
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

    if (rangeMat.at<float>(row_index, column_index) != FLT_MAX) {
      continue;
    }

    rangeMat.at<float>(row_index, column_index) = range;

    const int index = column_index + row_index * Horizon_SCAN;

    if (!imuAvailable) {
      output_points[index] = makePoint(q, p.intensity);
      continue;
    }

    const Eigen::Affine3d transform = calc_transform(p.time);

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
  const nav_msgs::Odometry & start_msg,
  const nav_msgs::Odometry & end_msg)
{
  return int(round(start_msg.pose.covariance[0])) == int(round(end_msg.pose.covariance[0]));
}

Eigen::Vector3d odomDeskewInfo(
  const nav_msgs::Odometry & start_msg,
  const nav_msgs::Odometry & end_msg)
{
  const Eigen::Affine3d begin = poseToAffine(start_msg.pose.pose);
  const Eigen::Affine3d end = poseToAffine(end_msg.pose.pose);

  const Eigen::Affine3d odom = begin.inverse() * end;

  return odom.translation();
}

void imuDeskewInfo(
  const double scan_start_time,
  const double scan_end_time,
  std::vector<double> & imuTime,
  std::vector<Eigen::Vector3d> & imuRot,
  std::deque<sensor_msgs::Imu> & imu_buffer,
  geometry_msgs::Vector3 & initialIMU)
{
  dropBefore(scan_start_time - 0.01, imu_buffer);

  if (imu_buffer.empty()) {
    return;
  }

  for (int i = 0; i < (int)imu_buffer.size(); ++i) {
    const double imu_time = timeInSec(imu_buffer[i].header);

    if (imu_time <= scan_start_time) {
      initialIMU = eigenToVector3(quaternionToRPY(imu_buffer[i].orientation));
    }

    if (imu_time > scan_end_time + 0.01) {
      break;
    }

    if (imuTime.size() == 0) {
      imuRot.push_back(Eigen::Vector3d::Zero());
      imuTime.push_back(imu_time);
      continue;
    }

    // get angular velocity
    const Eigen::Vector3d angular = imuAngular2rosAngular(imu_buffer[i].angular_velocity);

    // integrate rotation
    const Eigen::Vector3d rot = imuRot.back() + angular * (imu_time - imuTime.back());
    imuRot.push_back(rot);
    imuTime.push_back(imu_time);
  }
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
  const ros::Subscriber subLaserCloud;

  const ros::Publisher pubExtractedCloud;
  const ros::Publisher pubLaserCloudInfo;

  const ros::Subscriber subImu;
  const ros::Subscriber subOdom;

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
    const double scan_end_time = scan_start_time + input_cloud.points.back().time;

    if (!checkImuTime(imu_buffer, scan_start_time, scan_end_time)) {
      return;
    }

    lio_sam::cloud_info cloudInfo;
    cloudInfo.startRingIndex.assign(N_SCAN, 0);
    cloudInfo.endRingIndex.assign(N_SCAN, 0);

    cloudInfo.pointColInd.assign(N_SCAN * Horizon_SCAN, 0);
    cloudInfo.pointRange.assign(N_SCAN * Horizon_SCAN, 0);

    std::vector<double> imuTime;
    std::vector<Eigen::Vector3d> imuRot;

    bool imuAvailable = false;
    bool odomAvailable = false;

    Eigen::Vector3d odomInc = Eigen::Vector3d::Zero();

    {
      std::lock_guard<std::mutex> lock1(imuLock);
      std::lock_guard<std::mutex> lock2(odoLock);

      imuDeskewInfo(
        scan_start_time, scan_end_time, imuTime, imuRot, imu_buffer,
        cloudInfo.initialIMU);

      imuAvailable = imuTime.size() > 1;

      dropBefore(scan_start_time - 0.01, odomQueue);
      odomAvailable = odometryIsAvailable(odomQueue, scan_start_time, scan_end_time);

      if (odomAvailable) {
        const unsigned int start_index = indexNextTimeOf(odomQueue, scan_start_time);
        const nav_msgs::Odometry start_msg = odomQueue[start_index];

        const unsigned int end_index = indexNextTimeOf(odomQueue, scan_end_time);
        const nav_msgs::Odometry end_msg = odomQueue[end_index];

        cloudInfo.initial_pose = start_msg.pose.pose;

        if (doOdomDeskew(start_msg, end_msg)) {
          odomInc = odomDeskewInfo(start_msg, end_msg);
        }
      }
    }

    cloudInfo.odomAvailable = odomAvailable;
    cloudInfo.imuAvailable = imuAvailable;

    const AffineFinder calc_transform(imuRot, imuTime, odomInc, scan_start_time, scan_end_time);

    const auto [rangeMat, output_points] = projectPointCloud(
      input_cloud.points,
      range_min, range_max,
      downsampleRate, N_SCAN, Horizon_SCAN,
      cloudInfo.imuAvailable, calc_transform
    );

    pcl::PointCloud<PointType> extractedCloud;
    int count = 0;
    // extract segmented cloud for lidar odometry
    for (int i = 0; i < N_SCAN; ++i) {
      cloudInfo.startRingIndex[i] = count + 5;

      for (int j = 0; j < Horizon_SCAN; ++j) {
        const float range = rangeMat.at<float>(i, j);
        if (range == FLT_MAX) {
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
