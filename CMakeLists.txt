cmake_minimum_required(VERSION 3.10)
project(lio_sam)

set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O3 -Wall -g -pthread")

find_package(catkin REQUIRED COMPONENTS
  tf
  roscpp
  rospy
  cv_bridge
  eigen_conversions
  # pcl library
  pcl_conversions
  # msgs
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  message_generation
  visualization_msgs
)

find_package(OpenMP REQUIRED)
find_package(PCL REQUIRED QUIET)
find_package(GTSAM REQUIRED QUIET)
find_package(fmt REQUIRED)

add_message_files(
  DIRECTORY msg
  FILES
  cloud_info.msg
)

add_service_files(
  DIRECTORY srv
  FILES
  save_map.srv
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
  nav_msgs
  sensor_msgs
)

catkin_package(
  INCLUDE_DIRS include
  DEPENDS PCL GTSAM

  CATKIN_DEPENDS
  std_msgs
  nav_msgs
  geometry_msgs
  sensor_msgs
  message_runtime
  message_generation
  visualization_msgs
)

# include directories
include_directories(SYSTEM ${catkin_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} ${GTSAM_INCLUDE_DIR})

# link directories
link_directories(
	${PCL_LIBRARY_DIRS}
  ${GTSAM_LIBRARY_DIRS}
)

add_subdirectory(thirdparty/range-v3 EXCLUDE_FROM_ALL)

include_directories(include)
add_library(util SHARED
  src/cloud_optimizer.cpp
  src/param_server.cpp
  src/utility.cpp
)

target_compile_options(util PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(util ${catkin_LIBRARIES} gtsam fmt::fmt range-v3)

###########
## Build ##
###########

# Range Image Projection
add_executable(${PROJECT_NAME}_imageProjection src/imageProjection.cpp)
add_dependencies(${PROJECT_NAME}_imageProjection ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_link_libraries(
  ${PROJECT_NAME}_imageProjection ${catkin_LIBRARIES} ${PCL_LIBRARIES} util range-v3)

# Mapping Optimization
add_executable(${PROJECT_NAME}_mapOptmization src/mapOptmization.cpp)
add_dependencies(${PROJECT_NAME}_mapOptmization ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_compile_options(${PROJECT_NAME}_mapOptmization PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(
  ${PROJECT_NAME}_mapOptmization ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenMP_CXX_FLAGS} gtsam util range-v3)

# IMU Preintegration
add_executable(${PROJECT_NAME}_imuPreintegration src/imuPreintegration.cpp)
target_link_libraries(
  ${PROJECT_NAME}_imuPreintegration ${catkin_LIBRARIES} ${PCL_LIBRARIES} gtsam util range-v3)
