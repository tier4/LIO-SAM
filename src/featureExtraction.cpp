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
  std::vector<int> cloudSmoothness;
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

    cloudSmoothness.resize(N_SCAN * Horizon_SCAN);
  }

  void laserCloudInfoHandler(const lio_sam::cloud_infoConstPtr & msgIn)
  {
    std::vector<float> cloudCurvature(N_SCAN * Horizon_SCAN);
    std::vector<int> cloudNeighborPicked(N_SCAN * Horizon_SCAN);
    std::vector<int> cloudLabel(N_SCAN * Horizon_SCAN);

    pcl::PointCloud<PointType>::Ptr extractedCloud;
    pcl::PointCloud<PointType>::Ptr cornerCloud;
    pcl::PointCloud<PointType>::Ptr surfaceCloud;

    extractedCloud.reset(new pcl::PointCloud<PointType>());
    cornerCloud.reset(new pcl::PointCloud<PointType>());
    surfaceCloud.reset(new pcl::PointCloud<PointType>());

    pcl::VoxelGrid<PointType> downSizeFilter;

    downSizeFilter.setLeafSize(
      odometrySurfLeafSize, odometrySurfLeafSize,
      odometrySurfLeafSize);

    lio_sam::cloud_info cloudInfo = *msgIn; // new cloud info
    std_msgs::Header cloudHeader = msgIn->header; // new cloud header
    // new cloud for extraction
    pcl::fromROSMsg(msgIn->cloud_deskewed, *extractedCloud);

    const int cloudSize = extractedCloud->points.size();

    const std::vector<float> & range = cloudInfo.pointRange;
    for (int i = 5; i < cloudSize - 5; i++) {
      const float d = range[i - 5] + range[i - 4] + range[i - 3] + range[i - 2] + range[i - 1] -
        range[i] * 10 +
        range[i + 1] + range[i + 2] + range[i + 3] + range[i + 4] + range[i + 5];

      cloudCurvature[i] = d * d;
      cloudSmoothness[i] = i;
    }

    for (int i = 5; i < cloudSize - 5; i++) {
      cloudLabel[i] = 0;
    }

    for (int i = 5; i < cloudSize - 5; i++) {
      cloudNeighborPicked[i] = 0;
    }

    const std::vector<int> & column_index = cloudInfo.pointColInd;
    // mark occluded points and parallel beam points
    for (int i = 5; i < cloudSize - 6; ++i) {
      // occluded points
      float depth1 = cloudInfo.pointRange[i];
      float depth2 = cloudInfo.pointRange[i + 1];
      const int d = std::abs(int(column_index[i + 1] - column_index[i]));

      if (d < 10) {
        // 10 pixel diff in range image
        if (depth1 - depth2 > 0.3) {
          cloudNeighborPicked[i - 5] = 1;
          cloudNeighborPicked[i - 4] = 1;
          cloudNeighborPicked[i - 3] = 1;
          cloudNeighborPicked[i - 2] = 1;
          cloudNeighborPicked[i - 1] = 1;
          cloudNeighborPicked[i] = 1;
        } else if (depth2 - depth1 > 0.3) {
          cloudNeighborPicked[i + 1] = 1;
          cloudNeighborPicked[i + 2] = 1;
          cloudNeighborPicked[i + 3] = 1;
          cloudNeighborPicked[i + 4] = 1;
          cloudNeighborPicked[i + 5] = 1;
          cloudNeighborPicked[i + 6] = 1;
        }
      }
      // parallel beam
      float diff1 = std::abs(
        float(cloudInfo.pointRange[i - 1] -
        cloudInfo.pointRange[i]));
      float diff2 = std::abs(
        float(cloudInfo.pointRange[i + 1] -
        cloudInfo.pointRange[i]));

      if (diff1 > 0.02 * cloudInfo.pointRange[i] &&
        diff2 > 0.02 * cloudInfo.pointRange[i])
      {
        cloudNeighborPicked[i] = 1;
      }
    }

    cornerCloud->clear();
    surfaceCloud->clear();

    pcl::PointCloud<PointType>::Ptr surfaceCloudScanDS(
      new pcl::PointCloud<PointType>()
    );

    for (int i = 0; i < N_SCAN; i++) {
      pcl::PointCloud<PointType>::Ptr surfaceCloudScan(
        new pcl::PointCloud<PointType>()
      );
      surfaceCloudScan->clear();

      for (int j = 0; j < 6; j++) {

        int sp = (cloudInfo.startRingIndex[i] * (6 - j) +
          cloudInfo.endRingIndex[i] * j) / 6;
        int ep = (cloudInfo.startRingIndex[i] * (5 - j) +
          cloudInfo.endRingIndex[i] * (j + 1)) / 6 - 1;

        if (sp >= ep) {
          continue;
        }

        std::sort(
          cloudSmoothness.begin() + sp, cloudSmoothness.begin() + ep,
          by_value(cloudCurvature));

        int largestPickedNum = 0;
        for (int k = ep; k >= sp; k--) {
          int ind = cloudSmoothness[k];
          if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] > edgeThreshold) {
            largestPickedNum++;
            if (largestPickedNum <= 20) {
              cloudLabel[ind] = 1;
              cornerCloud->push_back(extractedCloud->points[ind]);
            } else {
              break;
            }

            cloudNeighborPicked[ind] = 1;
            for (int l = 1; l <= 5; l++) {
              const int d = std::abs(int(column_index[ind + l] - column_index[ind + l - 1]));
              if (d > 10) {
                break;
              }
              cloudNeighborPicked[ind + l] = 1;
            }
            for (int l = -1; l >= -5; l--) {
              const int d = std::abs(int(column_index[ind + l] - column_index[ind + l + 1]));
              if (d > 10) {
                break;
              }
              cloudNeighborPicked[ind + l] = 1;
            }
          }
        }

        for (int k = sp; k <= ep; k++) {
          int ind = cloudSmoothness[k];
          if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] < surfThreshold) {

            cloudLabel[ind] = -1;
            cloudNeighborPicked[ind] = 1;

            for (int l = 1; l <= 5; l++) {

              const int d = std::abs(int(column_index[ind + l] - column_index[ind + l - 1]));
              if (d > 10) {
                break;
              }

              cloudNeighborPicked[ind + l] = 1;
            }
            for (int l = -1; l >= -5; l--) {

              const int d = std::abs(int(column_index[ind + l] - column_index[ind + l + 1]));
              if (d > 10) {
                break;
              }

              cloudNeighborPicked[ind + l] = 1;
            }
          }
        }

        for (int k = sp; k <= ep; k++) {
          if (cloudLabel[k] <= 0) {
            surfaceCloudScan->push_back(extractedCloud->points[k]);
          }
        }
      }

      surfaceCloudScanDS->clear();
      downSizeFilter.setInputCloud(surfaceCloudScan);
      downSizeFilter.filter(*surfaceCloudScanDS);

      *surfaceCloud += *surfaceCloudScanDS;
    }

    // free cloud info memory
    cloudInfo.startRingIndex.clear();
    cloudInfo.endRingIndex.clear();
    cloudInfo.pointColInd.clear();
    cloudInfo.pointRange.clear();

    // save newly extracted features
    cloudInfo.cloud_corner = publishCloud(
      &pubCornerPoints, *cornerCloud,
      cloudHeader.stamp, lidarFrame);
    cloudInfo.cloud_surface = publishCloud(
      &pubSurfacePoints, *surfaceCloud,
      cloudHeader.stamp, lidarFrame);
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
