#pragma once

#include "common.h"


class EdgeCostFunction : public ceres::SizedCostFunction<3,7>
{
public:

    EdgeCostFunction(Eigen::Vector3d curr_point_, Eigen::Vector3d point_a_, Eigen::Vector3d point_b_) :
                    curr_point(curr_point_), point_a(point_a_), point_b(point_b_)
    {

    }

    virtual ~EdgeCostFunction()
    {

    }

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        Eigen::Map<const Eigen::Quaterniond> q_w_curr(parameters[0]);
        Eigen::Map<const Eigen::Vector3d> t_w_curr(parameters[0]+4);
        Eigen::Vector3d lp;
        lp = q_w_curr * curr_point + t_w_curr;  // 转换到世界坐标系

        Eigen::Vector3d nu = (lp - point_a).cross(lp - point_b);  // 叉乘结果为平行四面行面积
        Eigen::Vector3d ab = point_a - point_b;  // 底边向量
        double ab_norm = ab.norm();  // 底边长
        residuals[0] = nu.x() / ab_norm; 
        residuals[1] = nu.y() / ab_norm; 
        residuals[2] = nu.z() / ab_norm; 

        if(jacobians)
        {
            if(jacobians[0])
            {
                Eigen::Matrix3d skew_lp = skew(lp);
                Eigen::Matrix<double, 3, 6> J_se3;  // 旋转在前，平移在后
                J_se3.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
                J_se3.block<3, 3>(0, 0) = -skew_lp;

                Eigen::Map< Eigen::Matrix<double, 3, 7, Eigen::RowMajor> > jacobian_pose(jacobians[0]);
                jacobian_pose.setZero();
                Eigen::Matrix3d skew_ab = skew(ab);
                jacobian_pose.block<3, 6>(0, 0) = -skew_ab * J_se3 / ab_norm;
            }
        }

        return true;
    }


private:

    Eigen::Vector3d curr_point;
    Eigen::Vector3d point_a;
    Eigen::Vector3d point_b;

};


class SurfCostFunction : public ceres::SizedCostFunction<1, 7>
{
public:

    SurfCostFunction(Eigen::Vector3d curr_point_, Eigen::Vector3d norm_, double negative_OA_dot_norm_) :
                    curr_point(curr_point_), norm(norm_), negative_OA_dot_norm(negative_OA_dot_norm_)
    {

    }

    virtual ~SurfCostFunction()
    {

    }

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        Eigen::Map<const Eigen::Quaterniond> q_w_curr(parameters[0]);
        Eigen::Map<const Eigen::Vector3d> t_w_curr(parameters[0] + 4);
        Eigen::Vector3d point_w = q_w_curr * curr_point + t_w_curr;
        residuals[0] = norm.dot(point_w) + negative_OA_dot_norm;

        if(jacobians)
        {
            if(jacobians[0])
            {
                Eigen::Matrix3d skew_p = skew(point_w);
                Eigen::Matrix<double, 3, 6> J_se3;
                J_se3.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
                J_se3.block<3, 3>(0, 0) = -skew_p;

                Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor> > jacobian_pose(jacobians[0]);
                jacobian_pose.setZero();
                jacobian_pose.block<1, 6>(0, 0) = norm.transpose() * J_se3;
            }
        }

        return true;
    }


private:

    Eigen::Vector3d curr_point;
    Eigen::Vector3d norm;
    double negative_OA_dot_norm;

};