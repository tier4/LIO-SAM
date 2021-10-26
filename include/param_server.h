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
  std::string gpsTopic;

  //Frames
  std::string lidarFrame;
  std::string baselinkFrame;
  std::string odometryFrame;
  std::string mapFrame;

  // GPS Settings
  bool useImuHeadingInitialization;
  bool useGpsElevation;
  float gpsCovThreshold;
  float poseCovThreshold;

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
  std::vector < double > extTransV;
  Eigen::Vector3d extTrans;

  // LOAM
  float edgeThreshold;
  float surfThreshold;
  int edgeFeatureMinValidNum;
  int surfFeatureMinValidNum;

  // voxel filter paprams
  float surface_leaf_size;
  float mappingCornerLeafSize;
  float mappingSurfLeafSize;

  // CPU Params
  int numberOfCores;
  double mappingProcessInterval;

  // Surrounding map
  float surroundingkeyframeAddingDistThreshold;
  float surroundingkeyframeAddingAngleThreshold;
  float surroundingKeyframeDensity;
  float surroundingKeyframeSearchRadius;

  // Loop closure
  bool loopClosureEnableFlag;
  float loopClosureFrequency;
  int surroundingKeyframeSize;
  float historyKeyframeSearchRadius;
  float historyKeyframeSearchTimeDiff;
  int historyKeyframeSearchNum;
  float historyKeyframeFitnessScore;

  // global map visualization radius
  float globalMapVisualizationSearchRadius;
  float globalMapVisualizationPoseDensity;
  float globalMapVisualizationLeafSize;
};
