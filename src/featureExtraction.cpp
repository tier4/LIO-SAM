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
  Default = 0,
  Edge = 1,
  Surface = -1
};

void neighborPicked(
  const std::vector<int> & column_indices,
  const int index,
  std::vector<bool> & neighbor_picked)
{
  neighbor_picked[index] = true;
  for (int l = 1; l <= 5; l++) {
    const int d = std::abs(int(column_indices[index + l] - column_indices[index + l - 1]));
    if (d > 10) {
      break;
    }
    neighbor_picked[index + l] = true;
  }
  for (int l = -1; l >= -5; l--) {
    const int d = std::abs(int(column_indices[index + l] - column_indices[index + l + 1]));
    if (d > 10) {
      break;
    }
    neighbor_picked[index + l] = true;
  }
}

std::tuple<std::vector<float>, std::vector<int>>
calcCurvature(
  const pcl::PointCloud<PointType>::Ptr & points,
  const std::vector<float> & range,
  const int N_SCAN,
  const int Horizon_SCAN)
{
  std::vector<float> curvature(N_SCAN * Horizon_SCAN);
  std::vector<int> indices(N_SCAN * Horizon_SCAN, 0);
  for (unsigned int i = 5; i < points->size() - 5; i++) {
    const float d =
      range[i - 5] + range[i - 4] + range[i - 3] + range[i - 2] + range[i - 1] -
      range[i] * 10 +
      range[i + 1] + range[i + 2] + range[i + 3] + range[i + 4] + range[i + 5];

    curvature[i] = d * d;
    indices[i] = i;
  }
  return {curvature, indices};
}

class IndexRange
{
public:
  IndexRange(const int start_index, const int end_index, const int n_blocks)
  : start_index_(static_cast<double>(start_index)),
    end_index_(static_cast<double>(end_index)),
    n_blocks_(static_cast<double>(n_blocks))
  {
  }

  int begin(const int j) const
  {
    const double n = n_blocks_;
    return static_cast<int>(start_index_ * (1. - j / n) + end_index_ * j / n);
  }

  int end(const int j) const
  {
    const double n = n_blocks_;
    const int k = j + 1;
    return static_cast<int>(start_index_ * (1. - k / n) + end_index_ * k / n - 1.);
  }

private:
  const double start_index_;
  const double end_index_;
  const double n_blocks_;
};

class FeatureExtraction : public ParamServer
{

public:
  const ros::Publisher pubLaserCloudInfo;
  const ros::Publisher pubEdgePoints;
  const ros::Publisher pubSurfacePoints;
  const ros::Subscriber subLaserCloudInfo;

  FeatureExtraction()
  : pubLaserCloudInfo(nh.advertise<lio_sam::cloud_info>("lio_sam/feature/cloud_info", 1)),
    pubEdgePoints(nh.advertise<sensor_msgs::PointCloud2>("lio_sam/feature/cloud_edge", 1)),
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

    const auto points = getPointCloud<PointType>(msg->cloud_deskewed);

    for (unsigned int i = 5; i < points->size() - 5; i++) {
      neighbor_picked[i] = false;
    }

    const std::vector<float> & range = msg->point_range;

    const std::vector<int> & column_indices = msg->point_column_indices;
    // mark occluded points and parallel beam points
    for (unsigned int i = 5; i < points->size() - 6; ++i) {
      // occluded points
      const int d = std::abs(int(column_indices[i + 1] - column_indices[i]));

      // 10 pixel diff in range image
      if (d < 10 && range[i] - range[i + 1] > 0.3) {
        for (int j = 0; j <= 5; j++) {
          neighbor_picked[i - j] = true;
        }
      }

      if (d < 10 && range[i + 1] - range[i] > 0.3) {
        for (int j = 1; j <= 6; j++) {
          neighbor_picked[i + j] = true;
        }
      }
    }

    for (unsigned int i = 5; i < points->size() - 6; ++i) {
      // parallel beam
      const float ratio1 = std::abs(range[i - 1] - range[i]) / range[i];
      const float ratio2 = std::abs(range[i + 1] - range[i]) / range[i];

      if (ratio1 > 0.02 && ratio2 > 0.02) {
        neighbor_picked[i] = true;
      }
    }

    auto [curvature, indices] = calcCurvature(points, range, N_SCAN, Horizon_SCAN);

    pcl::PointCloud<PointType>::Ptr edge(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr surface(new pcl::PointCloud<PointType>());

    const int N_BLOCKS = 6;

    std::vector<CurvatureLabel> label(N_SCAN * Horizon_SCAN);
    for (unsigned int i = 5; i < points->size() - 5; i++) {
      label[i] = CurvatureLabel::Default;
    }

    for (int i = 0; i < N_SCAN; i++) {
      pcl::PointCloud<PointType>::Ptr surface_scan(new pcl::PointCloud<PointType>());

      const IndexRange index_range(msg->ring_start_indices[i], msg->end_ring_indices[i], N_BLOCKS);
      for (int j = 0; j < N_BLOCKS; j++) {
        const int sp = index_range.begin(j);
        const int ep = index_range.end(j);
        std::sort(indices.begin() + sp, indices.begin() + ep, by_value(curvature));

        int n_picked = 0;
        for (int k = ep; k >= sp; k--) {
          const int index = indices[k];
          if (neighbor_picked[index] || curvature[index] <= edgeThreshold) {
            continue;
          }

          if (n_picked >= 20) {
            break;
          }

          n_picked++;

          edge->push_back(points->at(index));
          label[index] = CurvatureLabel::Edge;

          neighborPicked(column_indices, index, neighbor_picked);
        }

        for (int k = sp; k <= ep; k++) {
          const int index = indices[k];
          if (neighbor_picked[index] || curvature[index] >= surfThreshold) {
            continue;
          }

          label[index] = CurvatureLabel::Surface;

          neighborPicked(column_indices, index, neighbor_picked);
        }

        for (int k = sp; k <= ep; k++) {
          if (label[k] == CurvatureLabel::Default || label[k] == CurvatureLabel::Edge) {
            surface_scan->push_back(points->at(k));
          }
        }
      }

      *surface += *downsample(surface_scan, surface_leaf_size);
    }

    const auto edge_downsampled = downsample(edge, mappingEdgeLeafSize);
    const auto surface_downsampled = downsample(surface, mappingSurfLeafSize);

    lio_sam::cloud_info cloud_info = *msg; // new cloud info
    // save newly extracted features
    cloud_info.cloud_edge = toRosMsg(*edge_downsampled, msg->header.stamp, lidarFrame);
    cloud_info.cloud_surface = toRosMsg(*surface_downsampled, msg->header.stamp, lidarFrame);
    // for visualization
    pubEdgePoints.publish(cloud_info.cloud_deskewed);
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
