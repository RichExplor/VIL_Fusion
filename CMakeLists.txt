cmake_minimum_required(VERSION 2.8.3)
project(sensor_fusion)

## Compile as C++11, supported in ROS Kinetic and newer
set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -pthread -w")

## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geometry_msgs
  message_generation
  message_runtime
  nav_msgs
  roscpp
  rospy
  sensor_msgs
  std_msgs
  tf
  roslib
)

find_package(OpenMP REQUIRED)
find_package(GTSAM REQUIRED)
find_package(Eigen3)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem program_options system)


message("----------Library version---------")
message("----GTSAM_VERSION: ${GTSAM_VERSION}")
message("----PCL_VERSION: ${PCL_VERSION}")
message("----OpenCV_VERSION: ${OpenCV_VERSION}")
message("----CERES_VERSION: ${CERES_VERSION}")

###########
## Build ##
###########
catkin_package()


include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${GTSAM_INCLUDE_DIR}
)


## Visual_inertial_lidar
file(GLOB visual_feature_files
    "src/visual_inertial_lidar/feature_tracker/parameters.cpp"
     "src/visual_inertial_lidar/feature_tracker/feature_tracker.cpp"
     "src/visual_inertial_lidar/feature_tracker/feature_tracker_node.cpp" 
     "src/visual_inertial_lidar/feature_tracker/camera_models/*.cc"
)

file(GLOB visual_feature_mask_files
    "src/visual_inertial_lidar/feature_tracker/parameters.cpp"
     "src/visual_inertial_lidar/feature_tracker/feature_tracker.cpp"
     "src/visual_inertial_lidar/feature_tracker/feature_tracker_node_mask.cpp" 
     "src/visual_inertial_lidar/feature_tracker/camera_models/*.cc"
)

file(GLOB visual_estimator_files
    "src/visual_inertial_lidar/vins_estimator/*.cpp"
    "src/visual_inertial_lidar/vins_estimator/factor/*.cpp"
    "src/visual_inertial_lidar/vins_estimator/initial/*.cpp"
    "src/visual_inertial_lidar/vins_estimator/utility/*.cpp"
)

file(GLOB pose_graph_files
    "src/visual_inertial_lidar/pose_graph/*.cpp"
    "src/visual_inertial_lidar/pose_graph/utility/*.cpp"
    "src/visual_inertial_lidar/pose_graph/ThirdParty/*.cpp"
    "src/visual_inertial_lidar/pose_graph/ThirdParty/DBoW/*.cpp"
    "src/visual_inertial_lidar/pose_graph/ThirdParty/DUtils/*.cpp"
    "src/visual_inertial_lidar/pose_graph/ThirdParty/DVision/*.cpp"
    "src/visual_inertial_lidar/feature_tracker/camera_models/*.cc"
)

###loop_Fusion
file(GLOB global_fusion_files
    "src/global_fusion/*.cpp"
)

####Visual_inertial_lidar

# visual and lidar feature
add_executable(${PROJECT_NAME}_feature ${visual_feature_files})
target_link_libraries(${PROJECT_NAME}_feature  ${catkin_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES} ${PCL_LIBRARIES})

# visual feature mask
add_executable(${PROJECT_NAME}_feature_mask ${visual_feature_mask_files})
target_link_libraries(${PROJECT_NAME}_feature_mask  ${catkin_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES} ${PCL_LIBRARIES})

# visual and lidar eatimator
add_executable(${PROJECT_NAME}_eatimator ${visual_estimator_files})
target_link_libraries(${PROJECT_NAME}_eatimator ${catkin_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES} ${PCL_LIBRARIES})

# pose graph
add_executable(${PROJECT_NAME}_pose_graph ${pose_graph_files})
target_link_libraries(${PROJECT_NAME}_pose_graph ${catkin_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES})


###Global_Fusion

add_executable(${PROJECT_NAME}_poseGraphOptimization ${global_fusion_files} )
target_compile_options(${PROJECT_NAME}_poseGraphOptimization 
  PRIVATE ${OpenMP_CXX_FLAGS}
)
target_link_libraries(${PROJECT_NAME}_poseGraphOptimization ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES} ${OpenCV_LIBRARIES} ${OpenMP_CXX_FLAGS} gtsam)