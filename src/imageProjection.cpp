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

class IMUBuffer
{
public:
  void imuHandler(const sensor_msgs::Imu::ConstPtr & imuMsg)
  {
    sensor_msgs::Imu thisImu = imu_converter_.imuConverter(*imuMsg);

    std::lock_guard<std::mutex> lock1(imuLock);
    imuQueue.push_back(thisImu);
    double roll, pitch, yaw;
    tf::Quaternion orientation;
    tf::quaternionMsgToTF(thisImu.orientation, orientation);
    tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
  }
  std::deque<sensor_msgs::Imu> imuQueue;

private:
  IMUConverter imu_converter_;
};

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
  sensor_msgs::PointCloud2 currentCloudMsg;

  std::array<double, queueLength> imuTime;
  std::array<double, queueLength> imuRotX;
  std::array<double, queueLength> imuRotY;
  std::array<double, queueLength> imuRotZ;

  int imuPointerCur;
  bool firstPointFlag;
  Eigen::Affine3f transStartInverse;

  pcl::PointCloud<PointXYZIRT>::Ptr laserCloudIn;
  pcl::PointCloud<PointType>::Ptr fullCloud;
  pcl::PointCloud<PointType>::Ptr extractedCloud;

  int deskewFlag;
  cv::Mat rangeMat;

  bool odomDeskewFlag;
  float odomIncreX;
  float odomIncreY;
  float odomIncreZ;

  lio_sam::cloud_info cloudInfo;
  double timeScanCur;
  double timeScanEnd;
  std_msgs::Header cloudHeader;
  IMUBuffer imu_buffer;

public:
  ImageProjection()
  : deskewFlag(0)
  {
    subImu = nh.subscribe(
      imuTopic, 2000,
      &IMUBuffer::imuHandler, &imu_buffer,
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
      imuRotX[i] = 0;
      imuRotY[i] = 0;
      imuRotZ[i] = 0;
    }

    pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
  }

  ~ImageProjection() {}

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

    if (!deskewInfo()) {
      return;
    }

    projectPointCloud(laserCloudIn->points);

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
      imuRotX[i] = 0;
      imuRotY[i] = 0;
      imuRotZ[i] = 0;
    }
  }

  bool cachePointCloud(const sensor_msgs::PointCloud2ConstPtr & laserCloudMsg)
  {

    // cache point cloud
    cloudQueue.push_back(*laserCloudMsg);
    if (cloudQueue.size() <= 2) {
      return false;
    }

    pcl::PointCloud<OusterPointXYZIRT> tmpOusterCloudIn;

    // convert cloud
    currentCloudMsg = std::move(cloudQueue.front());
    cloudQueue.pop_front();
    if (sensor == SensorType::VELODYNE) {
      pcl::moveFromROSMsg(currentCloudMsg, *laserCloudIn);
    } else if (sensor == SensorType::OUSTER) {
      // Convert to Velodyne format
      pcl::moveFromROSMsg(currentCloudMsg, tmpOusterCloudIn);
      laserCloudIn->points.resize(tmpOusterCloudIn.size());
      laserCloudIn->is_dense = tmpOusterCloudIn.is_dense;
      for (size_t i = 0; i < tmpOusterCloudIn.size(); i++) {
        auto & src = tmpOusterCloudIn.points[i];
        auto & dst = laserCloudIn->points[i];
        dst.x = src.x;
        dst.y = src.y;
        dst.z = src.z;
        dst.intensity = src.intensity;
        dst.ring = src.ring;
        dst.time = src.t * 1e-9f;
      }
    } else {
      ROS_ERROR_STREAM("Unknown sensor type: " << int(sensor));
      ros::shutdown();
    }

    // get timestamp
    cloudHeader = currentCloudMsg.header;
    timeScanCur = cloudHeader.stamp.toSec();
    timeScanEnd = timeScanCur + laserCloudIn->points.back().time;

    // check dense flag
    if (laserCloudIn->is_dense == false) {
      ROS_ERROR("Point cloud is not in dense format, please remove NaN points first!");
      ros::shutdown();
    }

    // check ring channel
    static int ringFlag = 0;
    if (ringFlag == 0) {
      ringFlag = -1;
      for (int i = 0; i < (int)currentCloudMsg.fields.size(); ++i) {
        if (currentCloudMsg.fields[i].name == "ring") {
          ringFlag = 1;
          break;
        }
      }
      if (ringFlag == -1) {
        ROS_ERROR("Point cloud ring channel not available, please configure your point cloud data!");
        ros::shutdown();
      }
    }

    // check point time
    if (deskewFlag == 0) {
      deskewFlag = -1;
      for (auto & field : currentCloudMsg.fields) {
        if (field.name == "time" || field.name == "t" || field.name == "time_stamp") {
          deskewFlag = 1;
          break;
        }
      }
      if (deskewFlag == -1) {
        ROS_WARN(
          "Point cloud timestamp not available, deskew function disabled, system will drift significantly!");
      }
    }

    return true;
  }

  bool deskewInfo()
  {
    std::lock_guard<std::mutex> lock1(imuLock);
    std::lock_guard<std::mutex> lock2(odoLock);

    // make sure IMU data available for the scan
    if (imu_buffer.imuQueue.empty()) {
      ROS_DEBUG("IMU queue empty ...");
      return false;
    }

    if (imu_buffer.imuQueue.front().header.stamp.toSec() > timeScanCur) {
      ROS_DEBUG("IMU time = %f", imu_buffer.imuQueue.front().header.stamp.toSec());
      ROS_DEBUG("LiDAR time = %f", timeScanCur);
      ROS_DEBUG("Timestamp of IMU data too late");
      return false;
    }

    if (imu_buffer.imuQueue.back().header.stamp.toSec() < timeScanEnd) {
      ROS_DEBUG("Timestamp of IMU data too early");
      return false;
    }

    imuDeskewInfo();

    odomDeskewInfo();

    return true;
  }

  void imuDeskewInfo()
  {
    cloudInfo.imuAvailable = false;

    while (!imu_buffer.imuQueue.empty()) {
      if (imu_buffer.imuQueue.front().header.stamp.toSec() < timeScanCur - 0.01) {
        imu_buffer.imuQueue.pop_front();
      } else {
        break;
      }
    }

    if (imu_buffer.imuQueue.empty()) {
      return;
    }

    imuPointerCur = 0;

    for (int i = 0; i < (int)imu_buffer.imuQueue.size(); ++i) {
      sensor_msgs::Imu thisImuMsg = imu_buffer.imuQueue[i];
      double currentImuTime = thisImuMsg.header.stamp.toSec();

      // get roll, pitch, and yaw estimation for this scan
      if (currentImuTime <= timeScanCur) {
        std::tie(
          cloudInfo.imuRollInit,
          cloudInfo.imuPitchInit,
          cloudInfo.imuYawInit) = imuRPY2rosRPY(thisImuMsg.orientation);
      }

      if (currentImuTime > timeScanEnd + 0.01) {
        break;
      }

      if (imuPointerCur == 0) {
        imuRotX[0] = 0;
        imuRotY[0] = 0;
        imuRotZ[0] = 0;
        imuTime[0] = currentImuTime;
        ++imuPointerCur;
        continue;
      }

      // get angular velocity
      const auto [angular_x, angular_y, angular_z] = imuAngular2rosAngular(
        thisImuMsg.angular_velocity);

      // integrate rotation
      double timeDiff = currentImuTime - imuTime[imuPointerCur - 1];
      imuRotX[imuPointerCur] = imuRotX[imuPointerCur - 1] + angular_x * timeDiff;
      imuRotY[imuPointerCur] = imuRotY[imuPointerCur - 1] + angular_y * timeDiff;
      imuRotZ[imuPointerCur] = imuRotZ[imuPointerCur - 1] + angular_z * timeDiff;
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
      if (odomQueue.front().header.stamp.toSec() < timeScanCur - 0.01) {
        odomQueue.pop_front();
      } else {
        break;
      }
    }

    if (odomQueue.empty()) {
      return;
    }

    if (odomQueue.front().header.stamp.toSec() > timeScanCur) {
      return;
    }

    // get start odometry at the beinning of the scan
    nav_msgs::Odometry startOdomMsg;

    for (int i = 0; i < (int)odomQueue.size(); ++i) {
      startOdomMsg = odomQueue[i];

      if (ROS_TIME(&startOdomMsg) < timeScanCur) {
        continue;
      } else {
        break;
      }
    }

    tf::Quaternion orientation;
    tf::quaternionMsgToTF(startOdomMsg.pose.pose.orientation, orientation);

    double roll, pitch, yaw;
    tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

    // Initial guess used in mapOptimization
    cloudInfo.initialGuessX = startOdomMsg.pose.pose.position.x;
    cloudInfo.initialGuessY = startOdomMsg.pose.pose.position.y;
    cloudInfo.initialGuessZ = startOdomMsg.pose.pose.position.z;
    cloudInfo.initialGuessRoll = roll;
    cloudInfo.initialGuessPitch = pitch;
    cloudInfo.initialGuessYaw = yaw;

    cloudInfo.odomAvailable = true;

    // get end odometry at the end of the scan
    odomDeskewFlag = false;

    if (odomQueue.back().header.stamp.toSec() < timeScanEnd) {
      return;
    }

    nav_msgs::Odometry endOdomMsg;

    for (int i = 0; i < (int)odomQueue.size(); ++i) {
      endOdomMsg = odomQueue[i];

      if (ROS_TIME(&endOdomMsg) < timeScanEnd) {
        continue;
      } else {
        break;
      }
    }

    if (int(round(startOdomMsg.pose.covariance[0])) != int(round(
        endOdomMsg.pose.covariance[0])))
    {
      return;
    }

    Eigen::Affine3f transBegin = pcl::getTransformation(
      startOdomMsg.pose.pose.position.x, startOdomMsg.pose.pose.position.y,
      startOdomMsg.pose.pose.position.z, roll, pitch, yaw);

    tf::quaternionMsgToTF(endOdomMsg.pose.pose.orientation, orientation);
    tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
    Eigen::Affine3f transEnd = pcl::getTransformation(
      endOdomMsg.pose.pose.position.x,
      endOdomMsg.pose.pose.position.y,
      endOdomMsg.pose.pose.position.z,
      roll, pitch, yaw);

    Eigen::Affine3f transBt = transBegin.inverse() * transEnd;

    float rollIncre, pitchIncre, yawIncre;
    pcl::getTranslationAndEulerAngles(
      transBt,
      odomIncreX, odomIncreY, odomIncreZ,
      rollIncre, pitchIncre, yawIncre);

    odomDeskewFlag = true;
  }

  std::tuple<float, float, float> findRotation(const double pointTime)
  {
    float rotXCur = 0;
    float rotYCur = 0;
    float rotZCur = 0;

    int imuPointerFront = 0;
    while (imuPointerFront < imuPointerCur) {
      if (pointTime < imuTime[imuPointerFront]) {
        break;
      }
      ++imuPointerFront;
    }

    if (pointTime > imuTime[imuPointerFront] || imuPointerFront == 0) {
      rotXCur = imuRotX[imuPointerFront];
      rotYCur = imuRotY[imuPointerFront];
      rotZCur = imuRotZ[imuPointerFront];
    } else {
      int imuPointerBack = imuPointerFront - 1;
      double ratioFront = (pointTime - imuTime[imuPointerBack]) /
        (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
      double ratioBack = (imuTime[imuPointerFront] - pointTime) /
        (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
      rotXCur = imuRotX[imuPointerFront] * ratioFront + imuRotX[imuPointerBack] * ratioBack;
      rotYCur = imuRotY[imuPointerFront] * ratioFront + imuRotY[imuPointerBack] * ratioBack;
      rotZCur = imuRotZ[imuPointerFront] * ratioFront + imuRotZ[imuPointerBack] * ratioBack;
    }
    return {rotXCur, rotYCur, rotZCur};
  }

  std::tuple<float, float, float> findPosition(double relTime)
  {
    float posXCur = 0;
    float posYCur = 0;
    float posZCur = 0;

    // If the sensor moves relatively slow, like walking speed,
    // positional deskew seems to have little benefits. Thus code below is commented.

    // if (cloudInfo.odomAvailable == false || odomDeskewFlag == false)
    //     return;

    // float ratio = relTime / (timeScanEnd - timeScanCur);

    // posXCur = ratio * odomIncreX;
    // posYCur = ratio * odomIncreY;
    // posZCur = ratio * odomIncreZ;
    return {posXCur, posYCur, posZCur};
  }

  PointType deskewPoint(const PointType & point, double relTime)
  {
    if (deskewFlag == -1 || cloudInfo.imuAvailable == false) {
      return point;
    }

    double pointTime = timeScanCur + relTime;

    const auto [rotXCur, rotYCur, rotZCur] = findRotation(pointTime);
    const auto [posXCur, posYCur, posZCur] = findPosition(relTime);

    const Eigen::Affine3f transform = pcl::getTransformation(
      posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur
    );

    if (firstPointFlag) {
      transStartInverse = transform.inverse();
      firstPointFlag = false;
    }

    // transform points to start
    Eigen::Affine3f transBt = transStartInverse * transform;

    Eigen::Vector3f p(point.x, point.y, point.z);

    const Eigen::Vector3f q = transBt * p;
    PointType newPoint;
    newPoint.x = q(0);
    newPoint.y = q(1);
    newPoint.z = q(2);
    newPoint.intensity = point.intensity;

    return newPoint;
  }

  void projectPointCloud(
    const std::vector<PointXYZIRT, Eigen::aligned_allocator<PointXYZIRT>> & points)
  {
    for (const PointXYZIRT & point : points) {
      PointType thisPoint;
      thisPoint.x = point.x;
      thisPoint.y = point.y;
      thisPoint.z = point.z;
      thisPoint.intensity = point.intensity;

      float range = pointDistance(thisPoint);
      if (range < lidarMinRange || range > lidarMaxRange) {
        continue;
      }

      int rowIdn = point.ring;
      if (rowIdn < 0 || rowIdn >= N_SCAN) {
        continue;
      }

      if (rowIdn % downsampleRate != 0) {
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

      if (rangeMat.at<float>(rowIdn, columnIdn) != FLT_MAX) {
        continue;
      }

      rangeMat.at<float>(rowIdn, columnIdn) = range;

      int index = columnIdn + rowIdn * Horizon_SCAN;
      fullCloud->points[index] = deskewPoint(thisPoint, point.time);
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
