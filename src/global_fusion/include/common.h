#pragma once

#include <vector>
#include <ctime>
#include <chrono>
#include <iostream>

#include <ros/ros.h>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <opencv/cv.h>
#include <opencv2/opencv.hpp>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

using namespace std;


typedef pcl::PointXYZI PointType;

struct Pose6D {
  double x;
  double y;
  double z;
  double roll;
  double pitch;
  double yaw;
};

struct keyFrameCloud
{
    Pose6D KeyFramePoses;
    Pose6D KeyFramePosesUpdated;
    double KeyFrameTimes;
};


inline double rad2deg(double radians)
{
    return radians * 180.0 / M_PI;
}

inline double deg2rad(double degrees)
{
    return degrees * M_PI / 180.0;
}

inline float pointDistance(PointType p)
{
    return sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
}

inline float DistanceXY(PointType p)
{
    return sqrt(p.x*p.x + p.y*p.y);
}

inline float poseDistance(Pose6D p)
{
    return sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
}

inline float pointDistance(PointType p1, PointType p2)
{
    return sqrt((p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y) + (p1.z-p2.z)*(p1.z-p2.z));
}

inline void coreImportTest (void)
{
    cout << "scancontext lib is successfully imported." << endl;
} 

inline float xy2theta( const float & _x, const float & _y )
{
    if ( _x >= 0 & _y >= 0) 
        return (180/M_PI) * atan(_y / _x);

    if ( _x < 0 & _y >= 0) 
        return 180 - ( (180/M_PI) * atan(_y / (-_x)) );

    if ( _x < 0 & _y < 0) 
        return 180 + ( (180/M_PI) * atan(_y / _x) );

    if ( _x >= 0 & _y < 0)
        return 360 - ( (180/M_PI) * atan((-_y) / _x) );
} 


Eigen::MatrixXd circshift( Eigen::MatrixXd &_mat, int _num_shift )
{
    // shift columns to right direction 
    assert(_num_shift >= 0);

    if( _num_shift == 0 )
    {
        Eigen::MatrixXd shifted_mat( _mat );
        return shifted_mat; // Early return 
    }

    Eigen::MatrixXd shifted_mat = Eigen::MatrixXd::Zero( _mat.rows(), _mat.cols() );
    for ( int col_idx = 0; col_idx < _mat.cols(); col_idx++ )
    {
        int new_location = (col_idx + _num_shift) % _mat.cols();
        shifted_mat.col(new_location) = _mat.col(col_idx);
    }

    return shifted_mat;
} 


std::vector<float> eig2stdvec( Eigen::MatrixXd _eigmat )
{
    std::vector<float> vec( _eigmat.data(), _eigmat.data() + _eigmat.size() );
    return vec;
} 


Eigen::Matrix3d skew(Eigen::Vector3d& vec)
{
    Eigen::Matrix3d mat = Eigen::Matrix3d::Zero();
    mat(0, 1) = -vec.z();
    mat(0, 2) =  vec.y();
    mat(1, 0) =  vec.z();
    mat(1, 2) = -vec.x();
    mat(2, 0) = -vec.y();
    mat(2, 1) =  vec.x();

    return mat;
}

void getTransformFromSe3(const Eigen::Matrix<double, 6, 1>& se3, Eigen::Quaterniond& q, Eigen::Vector3d& t)
{
    Eigen::Vector3d omega(se3.data());
    Eigen::Vector3d upsilon(se3.data() + 3);
    Eigen::Matrix3d Omega = skew(omega);

    double theta = omega.norm();
    double half_theta = 0.5*theta;

    double imag_factor;
    double real_factor = cos(half_theta);
    if(theta<1e-10)
    {
        double theta_sq = theta*theta;
        double theta_po4 = theta_sq*theta_sq;
        imag_factor = 0.5-0.0208333*theta_sq+0.000260417*theta_po4;
    }
    else
    {
        double sin_half_theta = sin(half_theta);
        imag_factor = sin_half_theta/theta;
    }

    q = Eigen::Quaterniond(real_factor, imag_factor*omega.x(), imag_factor*omega.y(), imag_factor*omega.z());


    Eigen::Matrix3d J;
    if (theta<1e-10)
    {
        J = q.matrix();
    }
    else
    {
        Eigen::Matrix3d Omega2 = Omega*Omega;
        J = (Eigen::Matrix3d::Identity() + (1-cos(theta))/(theta*theta)*Omega + (theta-sin(theta))/(pow(theta,3))*Omega2);
    }

    t = J*upsilon;

}

class TicToc
{
    public:

        TicToc()
        {
            tic();
        }

        void tic()
        {
            start = std::chrono::system_clock::now();
        }

        double toc()
        {
            end = std::chrono::system_clock::now();
            std::chrono::duration<double> elapsed_seconds = end  - start;
            return elapsed_seconds.count() * 1000;
        }
        

    private:
        std::chrono::time_point<std::chrono::system_clock> start, end;
};