#pragma once

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

using namespace std;
using namespace Eigen;

class lidarConstraintsBase
{

public:

    lidarConstraintsBase()
    {

    }

    void push_back(Eigen::Quaterniond temp_q, Eigen::Vector3d temp_t)
    {
        lidar_q = temp_q;
        lidar_t = temp_t;
    }

    Eigen::Quaterniond lidar_q;
    Eigen::Vector3d lidar_t;
};