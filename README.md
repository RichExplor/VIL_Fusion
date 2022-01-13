# Sensor_Fusion
## Multi-sensor fusion SLAM system based on factor graph

**说明：使用视觉-惯性和纯激光里程计，通过因子图后处理全局位姿。其中平移的权重激光比例高，旋转的权重视觉-惯性比例高**
**其中视觉前端特征点使用激光点云和三角化共同估计深度（添加mask-rcnn剔除动态特征点）, 激光前端采用帧到局部地图的特征最近邻检索匹配，后端优化计算雅克比矩阵，迭代优化，保证构建地图的精度**
**传感器类型：大恒工业相机、Xsens惯性传感器、VLP-32激光雷达、松灵机器人底盘**

This work is based on f-loam and VINS-Mono, On this basis, We use the laser point to obtain the depth and factor map and optimize the way to realize the pose estimation.

This code is modified from [F-LOAM](https://github.com/wh200720041/floam) and [VINS-Mono](https://github.com/HKUST-Aerial-Robotics/VINS-Mono) and [SC-A-LOAM]().

## 1.Working
  - [**visual_inertial**](#feature-depth-estimation)
  - [**lidar_fsloam**](#lidar_odometry)
  - [**global_fusion**](#graph-pose-estimation)
  - [**ScanContext loop detect**](#loop-detect)

## 2.Prerequisitis
  - [**Ubuntu-16.04**]
  - [**ROS-kinetic**]
  - [**Ceres-2.0.0**]
  - [**OpenCV-3.3.1**]
  - [**PCL-1.7.2**]
  - [**gtsam-4.0.2**]
  - [**Eigen-3.3.7**]
  - [**OpenMP**]

## 3.Build
### 3.1 clone build
```
    cd ~/catkin_ws/src
    git clone https://github.com/GuoFeng-X/Sensor_Fusion.git
    cd ..
    catkin_make -j4
```
### 3.2 run visual inertial
```
    cd catkin_ws
    source devel/setup.bash
    roslaunch sensor_fusion module_visual.launch
```
### 3.3 run fsloam and loop detect
```
    cd catkin_ws
    source devel/setup.bash
    roslaunch sensor_fusion module_fsloam.launch
```
```
    rosbag plag kitti_07.bag --pause --clock -r0.5
```
## 4. KITTI Demo
Using database KITTI-07 run this code.

<p align='center'>
<img width="65%" src="/img/kitti-08-全局轨迹.png"/>
</p>

## 5. Add Mask-RCNN
### 5.1 run mask-rcnn

**下载基于ROS话题发布的mask-rcnn前端代码。[mask-rcnn ros版本](https://download.csdn.net/download/qq_37568167/36765493)**

**在动态场景[ADVIO数据集](https://github.com/AaltoVision/ADVIO)下测试**

Copy the above code to the sensor_fusion folder

```
    cd catkin_ws/src/sensor_fusion
    unzip Mask-RCNN
    cd script/mask_rcnn
    ./run_build.sh
    cd detectron2/Visual_Frand
    ./run_detect_mask.sh
```

### 5.2 run visual estimation
```
    cd catkin_ws
    source devel/setup.bash
    roslaunch sensor_fusion module_visual_mask.launch
```

```
    rosbag plag ADVIO-05.bag --pause -r0.5
```

## 6.Acknowledgements
Thanks for [F-LOAM](https://github.com/wh200720041/floam) and [VINS-Mono](https://github.com/HKUST-Aerial-Robotics/VINS-Mono).

  - **F-LOAM(Wang H, Wang C, Chen C L, et al. F-loam: Fast lidar odometry and mapping.)**
  - **VINS-Mono(Tong Q, Peiliang L, et al. VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator.)**
