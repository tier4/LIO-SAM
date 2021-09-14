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

unsigned int indexNextTimeOf(const std::deque<nav_msgs::Odometry> & queue, const double time)
{
  for (unsigned int i = 0; i < queue.size(); ++i) {
    if (ROS_TIME(&queue[i]) < time) {
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
  const sensor_msgs::PointCloud2 & currentCloudMsg,
  const SensorType & sensor)
{
  pcl::PointCloud<PointXYZIRT> laserCloudIn;
  if (sensor == SensorType::VELODYNE) {
    pcl::fromROSMsg(currentCloudMsg, laserCloudIn);
    return laserCloudIn;
  }

  if (sensor == SensorType::OUSTER) {
    // Convert to Velodyne format
    pcl::PointCloud<OusterPointXYZIRT> tmpOusterCloudIn;

    pcl::fromROSMsg(currentCloudMsg, tmpOusterCloudIn);
    laserCloudIn.points.resize(tmpOusterCloudIn.size());
    laserCloudIn.is_dense = tmpOusterCloudIn.is_dense;
    for (size_t i = 0; i < tmpOusterCloudIn.size(); i++) {
      auto & src = tmpOusterCloudIn.points[i];
      auto & dst = laserCloudIn.points[i];
      dst.x = src.x;
      dst.y = src.y;
      dst.z = src.z;
      dst.intensity = src.intensity;
      dst.ring = src.ring;
      dst.time = src.t * 1e-9f;
    }
    return laserCloudIn;
  }

  throw std::runtime_error("Unknown sensor type");
}

Eigen::Vector3d findRotation(
  const std::array<Eigen::Vector3d, queueLength> & imuRot,
  const double point_time, const int imuPointerCur,
  const std::array<double, queueLength> & imuTime)
{
  int index = imuPointerCur;
  for (int i = 0; i < imuPointerCur; i++) {
    if (imuTime[i] > point_time) {
      index = i;
      break;
    }
  }

  if (point_time > imuTime[index] || index == 0) {
    return imuRot[index];
  }
  const Eigen::Vector3d prev_rot = imuRot[index - 1];
  const Eigen::Vector3d curr_rot = imuRot[index - 0];
  const double prev_time = imuTime[index - 1];
  const double curr_time = imuTime[index - 0];
  const double diff = imuTime[index] - imuTime[index - 1];
  return curr_rot * (point_time - prev_time) / diff + prev_rot * (curr_time - point_time) / diff;
}

Eigen::Vector3d findPosition(
  const Eigen::Vector3d & odomInc,
  const double timeScanCur,
  const double timeScanEnd,
  const double relTime,
  const bool odomAvailable,
  const bool odomDeskewFlag)
{
  const bool f = !odomAvailable || !odomDeskewFlag;
  const float ratio = relTime / (timeScanEnd - timeScanCur);
  const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
  return f ? zero : ratio * odomInc;
}

PointType deskewPoint(
  const Eigen::Vector3d & odomInc,
  const std::array<Eigen::Vector3d, queueLength> & imuRot,
  const std::array<double, queueLength> & imuTime,
  const double timeScanCur,
  const double timeScanEnd,
  const PointType & point,
  const double relTime,
  const int imuPointerCur,
  const bool odomDeskewFlag,
  const bool odomAvailable,
  bool & firstPointFlag,
  Eigen::Affine3d & transStartInverse)
{
  double pointTime = timeScanCur + relTime;

  const Eigen::Vector3d rotCur = findRotation(
    imuRot, pointTime, imuPointerCur, imuTime
  );
  const Eigen::Vector3d posCur = findPosition(
    odomInc, timeScanCur, timeScanEnd, relTime,
    odomAvailable, odomDeskewFlag
  );

  const Eigen::Affine3d transform = makeAffine(posCur, rotCur);

  if (firstPointFlag) {
    transStartInverse = transform.inverse();
    firstPointFlag = false;
  }

  // transform points to start
  const Eigen::Affine3d transBt = transStartInverse * transform;

  const Eigen::Vector3d p(point.x, point.y, point.z);

  const Eigen::Vector3d q = transBt * p;
  return makePoint(q, point.intensity);
}

void projectPointCloud(
  const Eigen::Vector3d & odomInc,
  const std::array<Eigen::Vector3d, queueLength> & imuRot,
  const std::array<double, queueLength> & imuTime,
  const Points<PointXYZIRT>::type & points,
  const int imuPointerCur,
  const bool odomDeskewFlag,
  const float lidarMinRange,
  const float lidarMaxRange,
  const double timeScanCur,
  const double timeScanEnd,
  const int downsampleRate,
  const int Horizon_SCAN,
  const bool imuAvailable,
  const bool odomAvailable,
  bool & firstPointFlag,
  cv::Mat & rangeMat,
  pcl::PointCloud<PointType>::Ptr & fullCloud,
  Eigen::Affine3d & transStartInverse)
{
  for (const PointXYZIRT & p : points) {
    const Eigen::Vector3d q(p.x, p.y, p.z);

    const float range = q.norm();
    if (range < lidarMinRange || lidarMaxRange < range) {
      continue;
    }

    const int row_index = p.ring;

    if (row_index % downsampleRate != 0) {
      continue;
    }

    const PointType point = makePoint(q, p.intensity);
    const float angle = rad2deg(atan2(point.x, point.y));
    const int f = static_cast<int>(Horizon_SCAN * (angle - 90.0) / 360.0);
    const int c = Horizon_SCAN / 2 - f;
    const int column_index = c % Horizon_SCAN;

    if (rangeMat.at<float>(row_index, column_index) != FLT_MAX) {
      continue;
    }

    rangeMat.at<float>(row_index, column_index) = range;

    const int index = column_index + row_index * Horizon_SCAN;

    if (!imuAvailable) {
      fullCloud->points[index] = point;
      continue;
    }

    fullCloud->points[index] = deskewPoint(
      odomInc, imuRot, imuTime, timeScanCur, timeScanEnd, point,
      p.time, imuPointerCur, odomDeskewFlag, odomAvailable,
      firstPointFlag, transStartInverse);
  }
}

void odomDeskewInfo(
  const double timeScanCur,
  const double timeScanEnd,
  bool & odomAvailable,
  geometry_msgs::Vector3 & initialXYZ,
  geometry_msgs::Vector3 & initialRPY,
  std::deque<nav_msgs::Odometry> & odomQueue,
  bool & odomDeskewFlag,
  Eigen::Vector3d & odomInc)
{
  while (!odomQueue.empty()) {
    if (timeInSec(odomQueue.front().header) < timeScanCur - 0.01) {
      odomQueue.pop_front();
    } else {
      break;
    }
  }

  if (odomQueue.empty()) {
    return;
  }

  if (timeInSec(odomQueue.front().header) > timeScanCur) {
    return;
  }

  // get start odometry at the beinning of the scan
  const unsigned int start_index = indexNextTimeOf(odomQueue, timeScanCur);
  const nav_msgs::Odometry startOdomMsg = odomQueue[start_index];

  const Eigen::Vector3d start_rpy = quaternionToRPY(startOdomMsg.pose.pose.orientation);
  const Eigen::Vector3d start_point = pointToEigen(startOdomMsg.pose.pose.position);
  // Initial guess used in mapOptimization
  initialXYZ = eigenToVector3(start_point);
  initialRPY = eigenToVector3(start_rpy);

  odomAvailable = true;

  // get end odometry at the end of the scan
  odomDeskewFlag = false;

  if (timeInSec(odomQueue.back().header) < timeScanEnd) {
    return;
  }

  const unsigned int end_index = indexNextTimeOf(odomQueue, timeScanEnd);
  const nav_msgs::Odometry endOdomMsg = odomQueue[end_index];

  if (int(round(startOdomMsg.pose.covariance[0])) != int(round(endOdomMsg.pose.covariance[0]))) {
    return;
  }

  const Eigen::Affine3d transBegin = makeAffine(start_point, start_rpy);

  const Eigen::Vector3d end_rpy = quaternionToRPY(endOdomMsg.pose.pose.orientation);
  const Eigen::Vector3d end_point = pointToEigen(endOdomMsg.pose.pose.position);
  const Eigen::Affine3d transEnd = makeAffine(end_point, end_rpy);

  const Eigen::Affine3d transBt = transBegin.inverse() * transEnd;

  odomInc = transBt.translation();
  odomDeskewFlag = true;
}

void imuDeskewInfo(
  const double timeScanCur,
  const double timeScanEnd,
  std::array<double, queueLength> & imuTime,
  std::array<Eigen::Vector3d, queueLength> & imuRot,
  std::deque<sensor_msgs::Imu> & imu_buffer,
  int & imuPointerCur,
  geometry_msgs::Vector3 & initialIMU,
  bool & imuAvailable)
{
  dropBefore(timeScanCur - 0.01, imu_buffer);

  if (imu_buffer.empty()) {
    return;
  }

  imuPointerCur = 0;

  for (int i = 0; i < (int)imu_buffer.size(); ++i) {
    const double currentImuTime = timeInSec(imu_buffer[i].header);

    // get roll, pitch, and yaw estimation for this scan
    if (currentImuTime <= timeScanCur) {
      const Eigen::Vector3d rpy = quaternionToRPY(imu_buffer[i].orientation);
      initialIMU.x = rpy(0);
      initialIMU.y = rpy(1);
      initialIMU.z = rpy(2);
    }

    if (currentImuTime > timeScanEnd + 0.01) {
      break;
    }

    if (imuPointerCur == 0) {
      imuRot[0] = Eigen::Vector3d::Zero();
      imuTime[0] = currentImuTime;
      ++imuPointerCur;
      continue;
    }

    // get angular velocity
    const Eigen::Vector3d angular = imuAngular2rosAngular(imu_buffer[i].angular_velocity);

    // integrate rotation
    double timeDiff = currentImuTime - imuTime[imuPointerCur - 1];
    imuRot[imuPointerCur] = imuRot[imuPointerCur - 1] + angular * timeDiff;
    imuTime[imuPointerCur] = currentImuTime;
    ++imuPointerCur;
  }

  --imuPointerCur;

  if (imuPointerCur <= 0) {
    return;
  }

  imuAvailable = true;
}

bool deskewInfo(
  const double timeScanCur,
  const double timeScanEnd,
  Eigen::Vector3d & odomInc,
  std::array<double, queueLength> & imuTime,
  std::array<Eigen::Vector3d, queueLength> & imuRot,
  std::deque<sensor_msgs::Imu> & imu_buffer,
  std::deque<nav_msgs::Odometry> & odomQueue,
  geometry_msgs::Vector3 & initialIMU,
  geometry_msgs::Vector3 & initialXYZ,
  geometry_msgs::Vector3 & initialRPY,
  bool & imuAvailable,
  bool & odomAvailable,
  bool & odomDeskewFlag,
  int & imuPointerCur)
{
  std::lock_guard<std::mutex> lock1(imuLock);
  std::lock_guard<std::mutex> lock2(odoLock);

  // make sure IMU data available for the scan
  if (imu_buffer.empty()) {
    ROS_DEBUG("IMU queue empty ...");
    return false;
  }

  if (timeInSec(imu_buffer.front().header) > timeScanCur) {
    ROS_DEBUG("IMU time = %f", timeInSec(imu_buffer.front().header));
    ROS_DEBUG("LiDAR time = %f", timeScanCur);
    ROS_DEBUG("Timestamp of IMU data too late");
    return false;
  }

  if (timeInSec(imu_buffer.back().header) < timeScanEnd) {
    ROS_DEBUG("Timestamp of IMU data too early");
    return false;
  }

  imuDeskewInfo(
    timeScanCur, timeScanEnd, imuTime, imuRot, imu_buffer, imuPointerCur,
    initialIMU, imuAvailable);

  odomDeskewInfo(
    timeScanCur, timeScanEnd,
    odomAvailable,
    initialXYZ,
    initialRPY,
    odomQueue, odomDeskewFlag, odomInc);

  return true;
}

class ImageProjection : public ParamServer
{
private:
  ros::Subscriber subLaserCloud;

  ros::Publisher pubExtractedCloud;
  ros::Publisher pubLaserCloudInfo;

  ros::Subscriber subImu;

  ros::Subscriber subOdom;
  std::deque<nav_msgs::Odometry> odomQueue;

  std::deque<sensor_msgs::PointCloud2> cloudQueue;

  std::array<double, queueLength> imuTime;
  std::array<Eigen::Vector3d, queueLength> imuRot;

  int imuPointerCur;
  bool firstPointFlag;
  Eigen::Affine3d transStartInverse;

  pcl::PointCloud<PointXYZIRT>::Ptr laserCloudIn;
  pcl::PointCloud<PointType>::Ptr fullCloud;
  pcl::PointCloud<PointType>::Ptr extractedCloud;

  cv::Mat rangeMat;

  bool odomDeskewFlag;
  Eigen::Vector3d odomInc;

  std::deque<sensor_msgs::Imu> imu_buffer;
  IMUConverter imu_converter_;

public:
  ImageProjection()
  {
    subImu = nh.subscribe(
      imuTopic, 2000,
      &ImageProjection::imuHandler, this,
      ros::TransportHints().tcpNoDelay());
    subOdom = nh.subscribe<nav_msgs::Odometry>(
      odomTopic + "_incremental", 2000,
      &ImageProjection::odometryHandler, this,
      ros::TransportHints().tcpNoDelay()
    );
    subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(
      pointCloudTopic, 5,
      &ImageProjection::cloudHandler, this, ros::TransportHints().tcpNoDelay());

    pubExtractedCloud =
      nh.advertise<sensor_msgs::PointCloud2>("lio_sam/deskew/cloud_deskewed", 1);
    pubLaserCloudInfo =
      nh.advertise<lio_sam::cloud_info>("lio_sam/deskew/cloud_info", 1);

    laserCloudIn.reset(new pcl::PointCloud<PointXYZIRT>());
    fullCloud.reset(new pcl::PointCloud<PointType>());
    extractedCloud.reset(new pcl::PointCloud<PointType>());

    fullCloud->points.resize(N_SCAN * Horizon_SCAN);

    laserCloudIn->clear();
    extractedCloud->clear();
    // reset range matrix for range image projection
    rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));

    imuPointerCur = 0;
    firstPointFlag = true;
    odomDeskewFlag = false;

    for (int i = 0; i < queueLength; ++i) {
      imuTime[i] = 0;
      imuRot[i] = Eigen::Vector3d::Zero();
    }

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

    // convert cloud
    const sensor_msgs::PointCloud2 currentCloudMsg = cloudQueue.front();
    cloudQueue.pop_front();

    try {
      *laserCloudIn = convert(currentCloudMsg, sensor);
    } catch (const std::runtime_error & e) {
      ROS_ERROR_STREAM("Unknown sensor type: " << int(sensor));
      ros::shutdown();
    }

    // check dense flag
    if (laserCloudIn->is_dense == false) {
      ROS_ERROR("Point cloud is not in dense format, please remove NaN points first!");
      ros::shutdown();
    }

    // check ring channel
    if (!ringIsAvailable(currentCloudMsg)) {
      ROS_ERROR(
        "Point cloud ring channel not available, "
        "please configure your point cloud data!"
      );
      ros::shutdown();
    }

    // check point time
    if (!timeStampIsAvailable(currentCloudMsg)) {
      ROS_ERROR(
        "Point cloud timestamp not available, deskew function disabled, "
        "system will drift significantly!"
      );
      ros::shutdown();
    }

    // get timestamp
    const std_msgs::Header cloudHeader = currentCloudMsg.header;
    const double timeScanCur = timeInSec(cloudHeader);
    const double timeScanEnd = timeScanCur + laserCloudIn->points.back().time;

    lio_sam::cloud_info cloudInfo;
    cloudInfo.startRingIndex.assign(N_SCAN, 0);
    cloudInfo.endRingIndex.assign(N_SCAN, 0);

    cloudInfo.pointColInd.assign(N_SCAN * Horizon_SCAN, 0);
    cloudInfo.pointRange.assign(N_SCAN * Horizon_SCAN, 0);

    bool imuAvailable = false;
    bool odomAvailable = false;

    const bool flag = deskewInfo(
      timeScanCur, timeScanEnd, odomInc,
      imuTime, imuRot, imu_buffer, odomQueue,
      cloudInfo.initialIMU, cloudInfo.initialXYZ, cloudInfo.initialRPY,
      imuAvailable, odomAvailable, odomDeskewFlag, imuPointerCur);

    if (!flag) {
      return;
    }

    cloudInfo.odomAvailable = odomAvailable;
    cloudInfo.imuAvailable = imuAvailable;

    projectPointCloud(
      odomInc, imuRot, imuTime, laserCloudIn->points,
      imuPointerCur, odomDeskewFlag,
      lidarMinRange, lidarMaxRange,
      timeScanCur, timeScanEnd,
      downsampleRate, Horizon_SCAN,
      cloudInfo.imuAvailable, cloudInfo.odomAvailable, firstPointFlag,
      rangeMat, fullCloud, transStartInverse
    );

    int count = 0;
    // extract segmented cloud for lidar odometry
    for (int i = 0; i < N_SCAN; ++i) {
      cloudInfo.startRingIndex[i] = count - 1 + 5;

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
        extractedCloud->push_back(fullCloud->points[j + i * Horizon_SCAN]);
        // size of extracted cloud
        ++count;
      }
      cloudInfo.endRingIndex[i] = count - 1 - 5;
    }

    cloudInfo.header = cloudHeader;
    cloudInfo.cloud_deskewed = publishCloud(
      &pubExtractedCloud, *extractedCloud,
      cloudHeader.stamp, lidarFrame);
    pubLaserCloudInfo.publish(cloudInfo);

    laserCloudIn->clear();
    extractedCloud->clear();
    // reset range matrix for range image projection
    rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));

    imuPointerCur = 0;
    firstPointFlag = true;
    odomDeskewFlag = false;

    for (int i = 0; i < queueLength; ++i) {
      imuTime[i] = 0;
      imuRot[i] = Eigen::Vector3d::Zero();
    }
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
