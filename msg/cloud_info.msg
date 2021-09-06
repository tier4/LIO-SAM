# Cloud Info
Header header 

int32[] startRingIndex
int32[] endRingIndex

int32[]  pointColInd # point column index in range image
float32[] pointRange # point range 

int64 imuAvailable
int64 odomAvailable

# Attitude for LOAM initialization
geometry_msgs/Vector3 initialIMU

# Initial guess from imu pre-integration
geometry_msgs/Vector3 initialXYZ
geometry_msgs/Vector3 initialRPY

# Point cloud messages
sensor_msgs/PointCloud2 cloud_deskewed  # original cloud deskewed
sensor_msgs/PointCloud2 cloud_corner    # extracted corner feature
sensor_msgs/PointCloud2 cloud_surface   # extracted surface feature