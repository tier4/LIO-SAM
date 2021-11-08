#include <ros/ros.h>
#include <Eigen/Core>

#include "matrix_type.h"

enum class SensorType {VELODYNE, OUSTER};

class ParamServer {
public:
  ParamServer();

  ros::NodeHandle nh;

  std::string robot_id;

  //Topics
  std::string pointCloudTopic;
  std::string imuTopic;
  std::string odomTopic;
  std::string imu_incremental_odometry_topic;

  //Frames
  std::string lidarFrame;
  std::string baselinkFrame;
  std::string odometryFrame;
  std::string mapFrame;

  // GPS Settings
  bool useImuHeadingInitialization;

  // Lidar Sensor Configuration
  SensorType sensor;
  int N_SCAN;
  int Horizon_SCAN;
  int downsampleRate;
  float range_min;
  float range_max;

  // IMU
  float imuAccNoise;
  float imuGyrNoise;
  float imuAccBiasN;
  float imuGyrBiasN;
  float imuGravity;
  float imuRPYWeight;
  Eigen::Vector3d extTrans;

  // LOAM
  float edgeThreshold;
  float surfThreshold;
  int min_edge_cloud;
  int min_surface_cloud;

  // voxel filter paprams
  float surface_leaf_size;
  float map_edge_leaf_size;
  float map_surface_leaf_size;

  // CPU Params
  int n_cores;
  double map_process_interval;

  // Surrounding map
  float keyframe_distance_threshold;
  float keyframe_angle_threshold;
  float keyframe_density;
  float keyframe_search_radius;
};
