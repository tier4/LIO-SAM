#include "utility.h"
#include "lio_sam/cloud_info.h"

struct smoothness_t
{
  float value;
  size_t ind;
};

class by_value
{
public:
  by_value(const std::vector<float> & values)
  : values_(values) {}
  bool operator()(const int & left, const int & right)
  {
    return values_[left] < values_[right];
  }

private:
  std::vector<float> values_;
};

class FeatureExtraction : public ParamServer
{

public:
  ros::Subscriber subLaserCloudInfo;

  ros::Publisher pubLaserCloudInfo;
  ros::Publisher pubCornerPoints;
  ros::Publisher pubSurfacePoints;
  std::vector<int> curvature_indices;
  FeatureExtraction()
  {
    subLaserCloudInfo =
      nh.subscribe<lio_sam::cloud_info>(
      "lio_sam/deskew/cloud_info", 1,
      &FeatureExtraction::laserCloudInfoHandler, this,
      ros::TransportHints().tcpNoDelay());

    pubLaserCloudInfo =
      nh.advertise<lio_sam::cloud_info>("lio_sam/feature/cloud_info", 1);
    pubCornerPoints =
      nh.advertise<sensor_msgs::PointCloud2>("lio_sam/feature/cloud_corner", 1);
    pubSurfacePoints =
      nh.advertise<sensor_msgs::PointCloud2>("lio_sam/feature/cloud_surface", 1);

    curvature_indices.resize(N_SCAN * Horizon_SCAN);
  }

  void laserCloudInfoHandler(const lio_sam::cloud_infoConstPtr & msgIn)
  {
    std::vector<bool> neighbor_picked(N_SCAN * Horizon_SCAN);
    std::vector<int> label(N_SCAN * Horizon_SCAN);

    pcl::VoxelGrid<PointType> downSizeFilter;

    downSizeFilter.setLeafSize(
      odometrySurfLeafSize, odometrySurfLeafSize,
      odometrySurfLeafSize);

    lio_sam::cloud_info cloudInfo = *msgIn; // new cloud info
    const ros::Time stamp = msgIn->header.stamp;

    const Points<PointType>::type points = getPointCloud<PointType>(msgIn->cloud_deskewed).points;

    for (int i = 5; i < points.size() - 5; i++) {
      label[i] = 0;
    }

    for (int i = 5; i < points.size() - 5; i++) {
      neighbor_picked[i] = false;
    }

    const std::vector<float> & range = cloudInfo.pointRange;

    const std::vector<int> & column_index = cloudInfo.pointColInd;
    // mark occluded points and parallel beam points
    for (int i = 5; i < points.size() - 6; ++i) {
      // occluded points
      const float depth1 = range[i];
      const float depth2 = range[i + 1];
      const int d = std::abs(int(column_index[i + 1] - column_index[i]));

      if (d < 10) {
        // 10 pixel diff in range image
        if (depth1 - depth2 > 0.3) {
          neighbor_picked[i - 5] = true;
          neighbor_picked[i - 4] = true;
          neighbor_picked[i - 3] = true;
          neighbor_picked[i - 2] = true;
          neighbor_picked[i - 1] = true;
          neighbor_picked[i - 0] = true;
        } else if (depth2 - depth1 > 0.3) {
          neighbor_picked[i + 1] = true;
          neighbor_picked[i + 2] = true;
          neighbor_picked[i + 3] = true;
          neighbor_picked[i + 4] = true;
          neighbor_picked[i + 5] = true;
          neighbor_picked[i + 6] = true;
        }
      }
      // parallel beam
      const float ratio1 = std::abs(float(range[i - 1] - range[i])) / range[i];
      const float ratio2 = std::abs(float(range[i + 1] - range[i])) / range[i];

      if (ratio1 > 0.02 && ratio2 > 0.02) {
        neighbor_picked[i] = true;
      }
    }

    std::vector<float> curvature(N_SCAN * Horizon_SCAN);
    for (int i = 5; i < points.size() - 5; i++) {
      const float d = range[i - 5] + range[i - 4] + range[i - 3] + range[i - 2] + range[i - 1] -
        range[i] * 10 +
        range[i + 1] + range[i + 2] + range[i + 3] + range[i + 4] + range[i + 5];

      curvature[i] = d * d;
      curvature_indices[i] = i;
    }

    pcl::PointCloud<PointType> corner;
    pcl::PointCloud<PointType> surface;

    const std::vector<int> & start_index = cloudInfo.startRingIndex;
    const std::vector<int> & end_index = cloudInfo.endRingIndex;
    for (int i = 0; i < N_SCAN; i++) {
      pcl::PointCloud<PointType>::Ptr surfaceCloudScan(new pcl::PointCloud<PointType>());

      for (int j = 0; j < 6; j++) {

        const int sp = (start_index[i] * (6 - j) + end_index[i] * j) / 6;
        const int ep = (start_index[i] * (5 - j) + end_index[i] * (j + 1)) / 6 - 1;

        if (sp >= ep) {
          continue;
        }

        std::sort(
          curvature_indices.begin() + sp, curvature_indices.begin() + ep,
          by_value(curvature));

        int largestPickedNum = 0;
        for (int k = ep; k >= sp; k--) {
          const int ind = curvature_indices[k];
          if (neighbor_picked[ind] || curvature[ind] <= edgeThreshold) {
            continue;
          }

          if (largestPickedNum >= 20) {
            break;
          }

          largestPickedNum++;

          label[ind] = 1;
          corner.push_back(points[ind]);

          neighbor_picked[ind] = true;
          for (int l = 1; l <= 5; l++) {
            const int d = std::abs(int(column_index[ind + l] - column_index[ind + l - 1]));
            if (d > 10) {
              break;
            }
            neighbor_picked[ind + l] = true;
          }
          for (int l = -1; l >= -5; l--) {
            const int d = std::abs(int(column_index[ind + l] - column_index[ind + l + 1]));
            if (d > 10) {
              break;
            }
            neighbor_picked[ind + l] = true;
          }
        }

        for (int k = sp; k <= ep; k++) {
          const int ind = curvature_indices[k];
          if (neighbor_picked[ind] || curvature[ind] >= surfThreshold) {
            continue;
          }

          label[ind] = -1;
          neighbor_picked[ind] = true;

          for (int l = 1; l <= 5; l++) {

            const int d = std::abs(int(column_index[ind + l] - column_index[ind + l - 1]));
            if (d > 10) {
              break;
            }

            neighbor_picked[ind + l] = true;
          }
          for (int l = -1; l >= -5; l--) {

            const int d = std::abs(int(column_index[ind + l] - column_index[ind + l + 1]));
            if (d > 10) {
              break;
            }

            neighbor_picked[ind + l] = true;
          }
        }

        for (int k = sp; k <= ep; k++) {
          if (label[k] <= 0) {
            surfaceCloudScan->push_back(points[k]);
          }
        }
      }

      downSizeFilter.setInputCloud(surfaceCloudScan);

      pcl::PointCloud<PointType> downsampled;
      downSizeFilter.filter(downsampled);

      surface += downsampled;
    }

    // free cloud info memory
    cloudInfo.startRingIndex.clear();
    cloudInfo.endRingIndex.clear();
    cloudInfo.pointColInd.clear();
    cloudInfo.pointRange.clear();

    // save newly extracted features
    cloudInfo.cloud_corner = publishCloud(&pubCornerPoints, corner, stamp, lidarFrame);
    cloudInfo.cloud_surface = publishCloud(&pubSurfacePoints, surface, stamp, lidarFrame);
    // publish to mapOptimization
    pubLaserCloudInfo.publish(cloudInfo);
  }
};


int main(int argc, char ** argv)
{
  ros::init(argc, argv, "lio_sam");

  FeatureExtraction FE;

  ROS_INFO("\033[1;32m----> Feature Extraction Started.\033[0m");

  ros::spin();

  return 0;
}
