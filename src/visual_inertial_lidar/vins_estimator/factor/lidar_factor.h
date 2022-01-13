#pragma once
#include <ros/assert.h>
#include <iostream>
#include <eigen3/Eigen/Dense>

#include "../utility/utility.h"
#include "../parameters.h"
#include "lidarConstraint_base.h"

#include <ceres/ceres.h>

class lidarFactor : public ceres::SizedCostFunction<6, 7, 7>
{
  public:
    lidarFactor() = delete;
    lidarFactor(lidarConstraintsBase* _lidarConstraints): lidarConstraints(_lidarConstraints)
    {
    }
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {

        Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

        Eigen::Vector3d Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
        Eigen::Quaterniond Qj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

        Eigen::Quaterniond qil(RIC[0] * RCL[0]);
        Eigen::Vector3d til = RIC[0] * TCL[0] + TIC[0];

        Eigen::Quaterniond qli = qil.inverse();
        Eigen::Vector3d tli = -(qil.inverse() * til);

        Eigen::Map<Eigen::Matrix<double, 6, 1>> residual(residuals);

        residual.block<3, 1>(O_P, 0) = qli * (Qi.inverse() * (Pj - Pi) - til - (qil * lidarConstraints->lidar_q * tli)) - lidarConstraints->lidar_t;
        residual.block<3, 1>(O_R, 0) = 2 * ((qil * lidarConstraints->lidar_q * qli).inverse() * (Qi.inverse() * Qj)).vec();

        Eigen::Matrix<double, 6, 6> sqrt_info = Eigen::Matrix<double, 6, 6>::Identity(); 
        sqrt_info.block<3, 3>(0, 0) = 1.0 / 0.1 * Eigen::Matrix3d::Identity();
        sqrt_info.block<3, 3>(3, 3) = 1.0 / 0.01 * Eigen::Matrix3d::Identity();
        residual = sqrt_info * residual;

        if (jacobians)
        {

            if (jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);
                jacobian_pose_i.setZero();

                jacobian_pose_i.block<3, 3>(O_P, O_P) = -(qli * Qi.inverse()).toRotationMatrix();
                jacobian_pose_i.block<3, 3>(O_P, O_R) = qli.toRotationMatrix() * Utility::skewSymmetric(Qi.inverse() * ( Pj - Pi));


                Eigen::Quaterniond corrected_delta_q = qil * lidarConstraints->lidar_q * qli;
                jacobian_pose_i.block<3, 3>(O_R, O_R) = -(Utility::Qleft(Qj.inverse() * Qi) * Utility::Qright(corrected_delta_q)).bottomRightCorner<3, 3>();

                if (jacobian_pose_i.maxCoeff() > 1e8 || jacobian_pose_i.minCoeff() < -1e8)
                {
                    ROS_WARN("numerical unstable in preintegration");
                }
            }
            if (jacobians[1])
            {
                Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[1]);
                jacobian_pose_j.setZero();

                jacobian_pose_j.block<3, 3>(O_P, O_P) = (qli * Qi.inverse()).toRotationMatrix();


                Eigen::Quaterniond corrected_delta_q = qil * lidarConstraints->lidar_q * qli;
                jacobian_pose_j.block<3, 3>(O_R, O_R) = Utility::Qleft(corrected_delta_q.inverse() * Qi.inverse() * Qj).bottomRightCorner<3, 3>();
            }
        }

        return true;
    }


    lidarConstraintsBase* lidarConstraints;

};

