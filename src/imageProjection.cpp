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

double timeInSec(const std_msgs::Header & header)
{
  return header.stamp.toSec();
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

pcl::PointCloud<PointXYZIRT> convert(
  sensor_msgs::PointCloud2 & currentCloudMsg,
  const SensorType & sensor)
{
  pcl::PointCloud<PointXYZIRT> laserCloudIn;
  if (sensor == SensorType::VELODYNE) {
    pcl::moveFromROSMsg(currentCloudMsg, laserCloudIn);
    return laserCloudIn;
  }

  if (sensor == SensorType::OUSTER) {
    // Convert to Velodyne format
    pcl::PointCloud<OusterPointXYZIRT> tmpOusterCloudIn;

    pcl::moveFromROSMsg(currentCloudMsg, tmpOusterCloudIn);
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

class ImageProjection : public ParamServer
{
private:
  std::mutex odoLock;

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

  bool deskewFlag;
  cv::Mat rangeMat;

  bool odomDeskewFlag;
  Eigen::Vector3d odomInc;

  lio_sam::cloud_info cloudInfo;
  double timeScanCur;
  double timeScanEnd;
  std_msgs::Header cloudHeader;

  std::deque<sensor_msgs::Imu> imu_buffer;
  IMUConverter imu_converter_;

public:
  ImageProjection()
  : deskewFlag(true)
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

    cloudInfo.startRingIndex.assign(N_SCAN, 0);
    cloudInfo.endRingIndex.assign(N_SCAN, 0);

    cloudInfo.pointColInd.assign(N_SCAN * Horizon_SCAN, 0);
    cloudInfo.pointRange.assign(N_SCAN * Horizon_SCAN, 0);

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
    if (!cachePointCloud(laserCloudMsg)) {
      return;
    }

    if (!deskewInfo(imuPointerCur)) {
      return;
    }

    projectPointCloud(laserCloudIn->points, imuPointerCur, odomDeskewFlag);

    int count = 0;
    // extract segmented cloud for lidar odometry
    for (int i = 0; i < N_SCAN; ++i) {
      cloudInfo.startRingIndex[i] = count - 1 + 5;

      for (int j = 0; j < Horizon_SCAN; ++j) {
        if (rangeMat.at<float>(i, j) != FLT_MAX) {
          // mark the points' column index for marking occlusion later
          cloudInfo.pointColInd[count] = j;
          // save range info
          cloudInfo.pointRange[count] = rangeMat.at<float>(i, j);
          // save extracted cloud
          extractedCloud->push_back(fullCloud->points[j + i * Horizon_SCAN]);
          // size of extracted cloud
          ++count;
        }
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

  bool cachePointCloud(const sensor_msgs::PointCloud2ConstPtr & laserCloudMsg)
  {

    // cache point cloud
    cloudQueue.push_back(*laserCloudMsg);
    if (cloudQueue.size() <= 2) {
      return false;
    }

    // convert cloud
    sensor_msgs::PointCloud2 currentCloudMsg = std::move(cloudQueue.front());
    cloudQueue.pop_front();
    try {
      *laserCloudIn = convert(currentCloudMsg, sensor);
    } catch (const std::runtime_error & e) {
      ROS_ERROR_STREAM("Unknown sensor type: " << int(sensor));
      ros::shutdown();
    }

    // get timestamp
    cloudHeader = currentCloudMsg.header;
    timeScanCur = timeInSec(cloudHeader);
    timeScanEnd = timeScanCur + laserCloudIn->points.back().time;

    // check dense flag
    if (laserCloudIn->is_dense == false) {
      ROS_ERROR("Point cloud is not in dense format, please remove NaN points first!");
      ros::shutdown();
    }

    // check ring channel
    if (!ringIsAvailable(currentCloudMsg)) {
      ROS_ERROR("Point cloud ring channel not available, please configure your point cloud data!");
      ros::shutdown();
    }

    // check point time
    if (deskewFlag) {
      deskewFlag = timeStampIsAvailable(currentCloudMsg);
      if (!deskewFlag) {
        ROS_WARN(
          "Point cloud timestamp not available, deskew function disabled, system will drift significantly!");
      }
    }

    return true;
  }

  bool deskewInfo(int & imuPointerCur)
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

    imuDeskewInfo(imuPointerCur);

    odomDeskewInfo();

    return true;
  }

  void imuDeskewInfo(int & imuPointerCur)
  {
    cloudInfo.imuAvailable = false;

    while (!imu_buffer.empty()) {
      if (timeInSec(imu_buffer.front().header) < timeScanCur - 0.01) {
        imu_buffer.pop_front();
      } else {
        break;
      }
    }

    if (imu_buffer.empty()) {
      return;
    }

    imuPointerCur = 0;

    for (int i = 0; i < (int)imu_buffer.size(); ++i) {
      double currentImuTime = timeInSec(imu_buffer[i].header);

      // get roll, pitch, and yaw estimation for this scan
      if (currentImuTime <= timeScanCur) {
        const Eigen::Vector3d rpy = quaternionToRPY(imu_buffer[i].orientation);
        cloudInfo.initialIMU.x = rpy(0);
        cloudInfo.initialIMU.y = rpy(1);
        cloudInfo.initialIMU.z = rpy(2);
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

    cloudInfo.imuAvailable = true;
  }

  void odomDeskewInfo()
  {
    cloudInfo.odomAvailable = false;

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
    cloudInfo.initialXYZ = eigenToVector3(start_point);
    cloudInfo.initialRPY = eigenToVector3(start_rpy);

    cloudInfo.odomAvailable = true;

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

  Eigen::Vector3d findRotation(const double pointTime, const int imuPointerCur) const
  {
    int index = 0;
    while (index < imuPointerCur) {
      if (pointTime < imuTime[index]) {
        break;
      }
      ++index;
    }

    if (pointTime > imuTime[index] || index == 0) {
      return imuRot[index];
    }

    const double diff = imuTime[index] - imuTime[index - 1];
    const double ratioFront = (pointTime - imuTime[index - 1]) / diff;
    const double ratioBack = (imuTime[index] - pointTime) / diff;
    return imuRot[index] * ratioFront + imuRot[index - 1] * ratioBack;
  }

  Eigen::Vector3d findPosition(
    const double relTime,
    const bool odomAvailable,
    const bool odomDeskewFlag) const
  {
    const bool f = !odomAvailable || !odomDeskewFlag;
    const float ratio = relTime / (timeScanEnd - timeScanCur);
    const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
    return f ? zero : ratio * odomInc;
  }

  PointType deskewPoint(
    const PointType & point, const double relTime,
    const int imuPointerCur, const bool odomDeskewFlag)
  {
    if (!deskewFlag || !cloudInfo.imuAvailable) {
      return point;
    }

    double pointTime = timeScanCur + relTime;

    const Eigen::Vector3d rotCur = findRotation(pointTime, imuPointerCur);
    const Eigen::Vector3d posCur = findPosition(relTime, cloudInfo.odomAvailable, odomDeskewFlag);

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
    const Points<PointXYZIRT>::type & points,
    const int imuPointerCur,
    const bool odomDeskewFlag)
  {
    for (const PointXYZIRT & point : points) {
      PointType thisPoint;
      thisPoint.x = point.x;
      thisPoint.y = point.y;
      thisPoint.z = point.z;
      thisPoint.intensity = point.intensity;

      float range = pointDistance(thisPoint);
      if (range < lidarMinRange || lidarMaxRange < range) {
        continue;
      }

      const int row_index = point.ring;
      if (row_index < 0 || N_SCAN <= row_index) {
        continue;
      }

      if (row_index % downsampleRate != 0) {
        continue;
      }

      float horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;

      static float ang_res_x = 360.0 / float(Horizon_SCAN);
      int columnIdn = -round((horizonAngle - 90.0) / ang_res_x) + Horizon_SCAN / 2;
      if (columnIdn >= Horizon_SCAN) {
        columnIdn -= Horizon_SCAN;
      }

      if (columnIdn < 0 || columnIdn >= Horizon_SCAN) {
        continue;
      }

      if (rangeMat.at<float>(row_index, columnIdn) != FLT_MAX) {
        continue;
      }

      rangeMat.at<float>(row_index, columnIdn) = range;

      int index = columnIdn + row_index * Horizon_SCAN;
      fullCloud->points[index] = deskewPoint(thisPoint, point.time, imuPointerCur, odomDeskewFlag);
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
