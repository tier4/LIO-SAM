#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>

template<typename T>
class KDTree
{
public:
  KDTree(const typename pcl::PointCloud<T>::Ptr & points)
  {
    kdtree_.setInputCloud(points);
  }

  std::tuple<std::vector<int>, std::vector<float>> radiusSearch(
    const T & point, const double radius) const
  {
    std::vector<int> indices;
    std::vector<float> squared_distances;
    kdtree_.radiusSearch(point, radius, indices, squared_distances);
    return {indices, squared_distances};
  }

  std::tuple<std::vector<int>, std::vector<float>> nearestKSearch(
    const T & point, const int k) const
  {
    std::vector<int> indices;
    std::vector<float> squared_distances;
    kdtree_.nearestKSearch(point, k, indices, squared_distances);
    return {indices, squared_distances};
  }

private:
  pcl::KdTreeFLANN<PointType> kdtree_;
};
