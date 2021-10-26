#include "param_server.h"

ParamServer::ParamServer()
{
  nh.param<std::string>("/robot_id", robot_id, "roboat");

  nh.param<std::string>(
    "lio_sam/pointCloudTopic", pointCloudTopic,
    "points_raw");
  nh.param<std::string>("lio_sam/imuTopic", imuTopic, "imu_correct");
  nh.param<std::string>("lio_sam/odomTopic", odomTopic, "odometry/imu");
  nh.param<std::string>("lio_sam/gpsTopic", gpsTopic, "odometry/gps");

  nh.param<std::string>("lio_sam/lidarFrame", lidarFrame, "base_link");
  nh.param<std::string>("lio_sam/baselinkFrame", baselinkFrame, "base_link");
  nh.param<std::string>("lio_sam/odometryFrame", odometryFrame, "odom");
  nh.param<std::string>("lio_sam/mapFrame", mapFrame, "map");

  nh.param<bool>(
    "lio_sam/useImuHeadingInitialization",
    useImuHeadingInitialization, false);
  nh.param<bool>("lio_sam/useGpsElevation", useGpsElevation, false);
  nh.param<float>("lio_sam/gpsCovThreshold", gpsCovThreshold, 2.0);
  nh.param<float>("lio_sam/poseCovThreshold", poseCovThreshold, 25.0);

  std::string sensorStr;
  nh.param<std::string>("lio_sam/sensor", sensorStr, "");
  if (sensorStr == "velodyne") {
    sensor = SensorType::VELODYNE;
  } else if (sensorStr == "ouster") {
    sensor = SensorType::OUSTER;
  } else {
    ROS_ERROR_STREAM(
      "Invalid sensor type (must be either 'velodyne' or 'ouster'): " << sensorStr);
    ros::shutdown();
  }

  nh.param<int>("lio_sam/N_SCAN", N_SCAN, 16);
  nh.param<int>("lio_sam/Horizon_SCAN", Horizon_SCAN, 1800);
  nh.param<int>("lio_sam/downsampleRate", downsampleRate, 1);
  nh.param<float>("lio_sam/lidarMinRange", range_min, 1.0);
  nh.param<float>("lio_sam/lidarMaxRange", range_max, 1000.0);

  nh.param<float>("lio_sam/imuAccNoise", imuAccNoise, 0.01);
  nh.param<float>("lio_sam/imuGyrNoise", imuGyrNoise, 0.001);
  nh.param<float>("lio_sam/imuAccBiasN", imuAccBiasN, 0.0002);
  nh.param<float>("lio_sam/imuGyrBiasN", imuGyrBiasN, 0.00003);
  nh.param<float>("lio_sam/imuGravity", imuGravity, 9.80511);
  nh.param<float>("lio_sam/imuRPYWeight", imuRPYWeight, 0.01);
  nh.param<std::vector<
      double>>("lio_sam/extrinsicTrans", extTransV, std::vector<double>());
  extTrans = Eigen::Map<const RowMajorMatrixXd>(extTransV.data(), 3, 1);

  nh.param<float>("lio_sam/edgeThreshold", edgeThreshold, 0.1);
  nh.param<float>("lio_sam/surfThreshold", surfThreshold, 0.1);
  nh.param<int>("lio_sam/edgeFeatureMinValidNum", edgeFeatureMinValidNum, 10);
  nh.param<int>("lio_sam/surfFeatureMinValidNum", surfFeatureMinValidNum, 100);

  nh.param<float>("lio_sam/odometrySurfLeafSize", surface_leaf_size, 0.2);
  nh.param<float>("lio_sam/mappingCornerLeafSize", mappingCornerLeafSize, 0.2);
  nh.param<float>("lio_sam/mappingSurfLeafSize", mappingSurfLeafSize, 0.2);

  nh.param<int>("lio_sam/numberOfCores", numberOfCores, 2);
  nh.param<double>(
    "lio_sam/mappingProcessInterval", mappingProcessInterval,
    0.15);

  nh.param<float>("lio_sam/keyframe_distance_threshold", keyframe_distance_threshold, 1.0);
  nh.param<float>("lio_sam/keyframe_angle_threshold", keyframe_angle_threshold, 0.2);
  nh.param<float>(
    "lio_sam/surroundingKeyframeDensity",
    surroundingKeyframeDensity, 1.0);
  nh.param<float>(
    "lio_sam/surroundingKeyframeSearchRadius",
    surroundingKeyframeSearchRadius, 50.0);

  nh.param<bool>("lio_sam/loopClosureEnableFlag", loopClosureEnableFlag, false);
  nh.param<float>("lio_sam/loopClosureFrequency", loopClosureFrequency, 1.0);
  nh.param<int>("lio_sam/surroundingKeyframeSize", surroundingKeyframeSize, 50);
  nh.param<float>(
    "lio_sam/historyKeyframeSearchRadius",
    historyKeyframeSearchRadius, 10.0);
  nh.param<float>(
    "lio_sam/historyKeyframeSearchTimeDiff",
    historyKeyframeSearchTimeDiff, 30.0);
  nh.param<int>(
    "lio_sam/historyKeyframeSearchNum", historyKeyframeSearchNum,
    25);
  nh.param<float>(
    "lio_sam/historyKeyframeFitnessScore",
    historyKeyframeFitnessScore, 0.3);

  nh.param<float>(
    "lio_sam/globalMapVisualizationSearchRadius",
    globalMapVisualizationSearchRadius, 1e3);
  nh.param<float>(
    "lio_sam/globalMapVisualizationPoseDensity",
    globalMapVisualizationPoseDensity, 10.0);
  nh.param<float>(
    "lio_sam/globalMapVisualizationLeafSize",
    globalMapVisualizationLeafSize, 1.0);

  usleep(100);
}
