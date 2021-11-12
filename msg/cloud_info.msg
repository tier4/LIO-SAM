# Cloud Info
Header header

int64 imu_orientation_available
int64 imu_odometry_available

# Attitude for LOAM initialization
geometry_msgs/Vector3 imu_orientation

# Initial guess from imu pre-integration
geometry_msgs/Pose scan_start_imu_pose

# Point cloud messages
sensor_msgs/PointCloud2 cloud_edge    # extracted edge feature
sensor_msgs/PointCloud2 cloud_surface   # extracted surface feature
