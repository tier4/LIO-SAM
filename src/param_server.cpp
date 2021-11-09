#include "param_server.h"

ParamServer::ParamServer()
{
  nh.param<std::string>("/robot_id", robot_id, "roboat");

  nh.param<std::string>("lio_sam/pointCloudTopic", pointCloudTopic, "points_raw");
  nh.param<std::string>("lio_sam/imuTopic", imuTopic, "imu_correct");
  nh.param<std::string>("lio_sam/odomTopic", odomTopic, "odometry/imu");
  imu_incremental_odometry_topic = odomTopic + "_incremental";
  nh.param<std::string>("lio_sam/lidarFrame", lidarFrame, "base_link");
  nh.param<std::string>("lio_sam/baselinkFrame", baselinkFrame, "base_link");
  nh.param<std::string>("lio_sam/odometryFrame", odometryFrame, "odom");
  nh.param<std::string>("lio_sam/mapFrame", mapFrame, "map");

  nh.param<bool>("lio_sam/useImuHeadingInitialization", useImuHeadingInitialization, false);

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

  std::vector<double> v;
  nh.param<std::vector<double>>("lio_sam/extrinsicTrans", v, std::vector<double>());
  extTrans = Eigen::Map<const RowMajorMatrixXd>(v.data(), 3, 1);

  std::vector<double> extRotV;
  nh.param<std::vector<double>>("lio_sam/extrinsicRot", extRotV, std::vector<double>());
  extRot = Eigen::Map<const RowMajorMatrixXd>(extRotV.data(), 3, 3);

  std::vector<double> extRPYV;
  nh.param<std::vector<double>>("lio_sam/extrinsicRPY", extRPYV, std::vector<double>());
  Eigen::Matrix3d extRPY = Eigen::Map<const RowMajorMatrixXd>(extRPYV.data(), 3, 3);
  extQRPY = Eigen::Quaterniond(extRPY);

  nh.param<float>("lio_sam/edgeThreshold", edgeThreshold, 0.1);
  nh.param<float>("lio_sam/surfThreshold", surfThreshold, 0.1);
  nh.param<int>("lio_sam/edgeFeatureMinValidNum", min_edge_cloud, 10);
  nh.param<int>("lio_sam/surfFeatureMinValidNum", min_surface_cloud, 100);

  nh.param<float>("lio_sam/odometrySurfLeafSize", surface_leaf_size, 0.2);
  nh.param<float>("lio_sam/mappingEdgeLeafSize", map_edge_leaf_size, 0.2);
  nh.param<float>("lio_sam/mappingSurfLeafSize", map_surface_leaf_size, 0.2);

  nh.param<int>("lio_sam/numberOfCores", n_cores, 2);
  nh.param<double>("lio_sam/mappingProcessInterval", map_process_interval, 0.15);

  nh.param<float>("lio_sam/keyframe_distance_threshold", keyframe_distance_threshold, 1.0);
  nh.param<float>("lio_sam/keyframe_angle_threshold", keyframe_angle_threshold, 0.2);
  nh.param<float>("lio_sam/surroundingKeyframeDensity", keyframe_density, 1.0);
  nh.param<float>("lio_sam/surroundingKeyframeSearchRadius", keyframe_search_radius, 50.0);

  usleep(100);
}
