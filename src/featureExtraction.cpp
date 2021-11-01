#include <pcl/filters/voxel_grid.h>

#include "message.hpp"
#include "downsample.hpp"
#include "utility.hpp"
#include "param_server.h"
#include "lio_sam/cloud_info.h"

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

enum class CurvatureLabel
{
  kDefault = 0,
  kEdge = 1,
  kSurface = -1
};

class FeatureExtraction : public ParamServer
{

public:
  const ros::Publisher pubLaserCloudInfo;
  const ros::Publisher pubCornerPoints;
  const ros::Publisher pubSurfacePoints;
  const ros::Subscriber subLaserCloudInfo;

  FeatureExtraction()
  : pubLaserCloudInfo(nh.advertise<lio_sam::cloud_info>("lio_sam/feature/cloud_info", 1)),
    pubCornerPoints(nh.advertise<sensor_msgs::PointCloud2>("lio_sam/feature/cloud_corner", 1)),
    pubSurfacePoints(nh.advertise<sensor_msgs::PointCloud2>("lio_sam/feature/cloud_surface", 1)),
    subLaserCloudInfo(
      nh.subscribe<lio_sam::cloud_info>(
        "lio_sam/deskew/cloud_info", 1,
        &FeatureExtraction::laserCloudInfoHandler, this,
        ros::TransportHints().tcpNoDelay()))
  {
  }

  void laserCloudInfoHandler(const lio_sam::cloud_infoConstPtr & msg) const
  {
    // used to prevent from labeling a neighbor as surface or edge
    std::vector<bool> neighbor_picked(N_SCAN * Horizon_SCAN);
    std::vector<CurvatureLabel> label(N_SCAN * Horizon_SCAN);

    lio_sam::cloud_info cloud_info = *msg; // new cloud info

    const auto points = getPointCloud<PointType>(msg->cloud_deskewed);

    for (unsigned int i = 5; i < points->size() - 5; i++) {
      label[i] = CurvatureLabel::kDefault;
    }

    for (unsigned int i = 5; i < points->size() - 5; i++) {
      neighbor_picked[i] = false;
    }

    const std::vector<float> & range = cloud_info.point_range;

    const std::vector<int> & column_index = cloud_info.point_column_indices;
    // mark occluded points and parallel beam points
    for (unsigned int i = 5; i < points->size() - 6; ++i) {
      // occluded points
      const int d = std::abs(int(column_index[i + 1] - column_index[i]));

      // 10 pixel diff in range image
      if (d < 10 && range[i] - range[i + 1] > 0.3) {
        neighbor_picked[i - 5] = true;
        neighbor_picked[i - 4] = true;
        neighbor_picked[i - 3] = true;
        neighbor_picked[i - 2] = true;
        neighbor_picked[i - 1] = true;
        neighbor_picked[i - 0] = true;
      }

      if (d < 10 && range[i + 1] - range[i] > 0.3) {
        neighbor_picked[i + 1] = true;
        neighbor_picked[i + 2] = true;
        neighbor_picked[i + 3] = true;
        neighbor_picked[i + 4] = true;
        neighbor_picked[i + 5] = true;
        neighbor_picked[i + 6] = true;
      }

      // parallel beam
      const float ratio1 = std::abs(float(range[i - 1] - range[i])) / range[i];
      const float ratio2 = std::abs(float(range[i + 1] - range[i])) / range[i];

      if (ratio1 > 0.02 && ratio2 > 0.02) {
        neighbor_picked[i] = true;
      }
    }

    std::vector<float> curvature(N_SCAN * Horizon_SCAN);
    std::vector<int> indices(N_SCAN * Horizon_SCAN, 0);
    for (unsigned int i = 5; i < points->size() - 5; i++) {
      const float d = range[i - 5] + range[i - 4] + range[i - 3] + range[i - 2] + range[i - 1] -
        range[i] * 10 +
        range[i + 1] + range[i + 2] + range[i + 3] + range[i + 4] + range[i + 5];

      curvature[i] = d * d;
      indices[i] = i;
    }

    pcl::PointCloud<PointType>::Ptr corner(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr surface(new pcl::PointCloud<PointType>());

    const std::vector<int> & start_indices = cloud_info.ring_start_indices;
    const std::vector<int> & end_indices = cloud_info.end_ring_indices;
    const int N_BLOCKS = 6;

    for (int i = 0; i < N_SCAN; i++) {
      pcl::PointCloud<PointType>::Ptr surface_scan(new pcl::PointCloud<PointType>());

      const int start_index = start_indices[i];
      const int end_index = end_indices[i];
      for (int j = 0; j < N_BLOCKS; j++) {

        const double n = static_cast<double>(N_BLOCKS);
        const int k = j + 1;
        const int sp = static_cast<int>(start_index * (1. - j / n) + end_index * j / n);
        const int ep = static_cast<int>(start_index * (1. - k / n) + end_index * k / n - 1.);

        if (sp >= ep) {
          continue;
        }

        std::sort(indices.begin() + sp, indices.begin() + ep, by_value(curvature));

        int largestPickedNum = 0;
        for (int k = ep; k >= sp; k--) {
          const int index = indices[k];
          if (neighbor_picked[index] || curvature[index] <= edgeThreshold) {
            continue;
          }

          if (largestPickedNum >= 20) {
            break;
          }

          largestPickedNum++;

          label[index] = CurvatureLabel::kEdge;
          corner->push_back(points->at(index));

          neighbor_picked[index] = true;
          for (int l = 1; l <= 5; l++) {
            const int d = std::abs(int(column_index[index + l] - column_index[index + l - 1]));
            if (d > 10) {
              break;
            }
            neighbor_picked[index + l] = true;
          }
          for (int l = -1; l >= -5; l--) {
            const int d = std::abs(int(column_index[index + l] - column_index[index + l + 1]));
            if (d > 10) {
              break;
            }
            neighbor_picked[index + l] = true;
          }
        }

        for (int k = sp; k <= ep; k++) {
          const int index = indices[k];
          if (neighbor_picked[index] || curvature[index] >= surfThreshold) {
            continue;
          }

          label[index] = CurvatureLabel::kSurface;
          neighbor_picked[index] = true;

          for (int l = 1; l <= 5; l++) {

            const int d = std::abs(int(column_index[index + l] - column_index[index + l - 1]));
            if (d > 10) {
              break;
            }

            neighbor_picked[index + l] = true;
          }
          for (int l = -1; l >= -5; l--) {

            const int d = std::abs(int(column_index[index + l] - column_index[index + l + 1]));
            if (d > 10) {
              break;
            }

            neighbor_picked[index + l] = true;
          }
        }

        for (int k = sp; k <= ep; k++) {
          if (label[k] == CurvatureLabel::kDefault || label[k] == CurvatureLabel::kEdge) {
            surface_scan->push_back(points->at(k));
          }
        }
      }

      *surface += *downsample(surface_scan, surface_leaf_size);
    }

    const auto corner_downsampled = downsample(corner, mappingCornerLeafSize);
    const auto surface_downsampled = downsample(surface, mappingSurfLeafSize);

    // save newly extracted features
    cloud_info.cloud_corner = toRosMsg(*corner_downsampled, msg->header.stamp, lidarFrame);
    cloud_info.cloud_surface = toRosMsg(*surface_downsampled, msg->header.stamp, lidarFrame);
    // for visualization
    pubCornerPoints.publish(cloud_info.cloud_deskewed);
    pubSurfacePoints.publish(cloud_info.cloud_surface);
    // publish to mapOptimization
    pubLaserCloudInfo.publish(cloud_info);
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
