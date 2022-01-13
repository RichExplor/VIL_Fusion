#pragma once

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/crop_box.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>

#include <pcl_conversions/pcl_conversions.h>

#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>

#include "lidarFactor.hpp"

class LocalSE3Parameterization : public ceres::LocalParameterization
{
public:

    LocalSE3Parameterization() 
    {

    }

    virtual ~LocalSE3Parameterization()
    {

    }

    virtual bool Plus(const double* x, const double* delta, double* x_plus_delta) const
    {
        Eigen::Map<const Eigen::Vector3d> trans(x + 4);

        Eigen::Quaterniond delta_q;
        Eigen::Vector3d delta_t;
        getTransformFromSe3(Eigen::Map<const Eigen::Matrix<double, 6, 1>>(delta), delta_q, delta_t);
        Eigen::Map<const Eigen::Quaterniond> quater(x);
        Eigen::Map<Eigen::Quaterniond> quater_plus(x_plus_delta);
        Eigen::Map<Eigen::Vector3d> trans_plus(x_plus_delta + 4);

        quater_plus = delta_q * quater;
        trans_plus = delta_q * trans + delta_t;

        return true;
    }

    virtual bool ComputeJacobian(const double* x, double* jacobian) const
    {
        Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> j(jacobian);
        (j.topRows(6)).setIdentity();
        (j.bottomRows(1)).setZero();

        return true;
    }

    virtual int GlobalSize() const
    {
        return 7;
    }

    virtual int LocalSize() const
    {
        return 6;
    }
};

class EstimationMapping
{

public:
    EstimationMapping()
    {
        
    }

    void initParameter(ros::NodeHandle& nh)
    {
        nh.param<double>("/EdgeLeafSize", edgeMapLeafSize, 0.2);
        nh.param<double>("/SurfLeafSize", surfMapLeafSize, 0.4);
        
        voxelEdgeFilter.setLeafSize(edgeMapLeafSize, edgeMapLeafSize, edgeMapLeafSize);
        voxelSurfFilter.setLeafSize(surfMapLeafSize, surfMapLeafSize, surfMapLeafSize);

        globalOdom = Eigen::Isometry3d::Identity();
        globalOdom_last = Eigen::Isometry3d::Identity();

        allocateMemory();
    }

    void allocateMemory()
    {
        cloudRegistered.reset(new pcl::PointCloud<PointType>());
        cloudNoRegistered.reset(new pcl::PointCloud<PointType>());
        localMapEdge.reset(new pcl::PointCloud<PointType>());
        localMapSurf.reset(new pcl::PointCloud<PointType>());

        kdtreeEdgeMap.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeSurfMap.reset(new pcl::KdTreeFLANN<PointType>());
    }

    void localMapInited(const pcl::PointCloud<PointType>::Ptr& edge_cloud, const pcl::PointCloud<PointType>::Ptr& surf_cloud)
    {
        *localMapEdge += *edge_cloud;
        *localMapSurf += *surf_cloud;

        *cloudRegistered += *edge_cloud;
        *cloudRegistered += *surf_cloud;

        *cloudNoRegistered += *edge_cloud;
        *cloudNoRegistered += *surf_cloud;
    }

    void EdgeCostFactor(const pcl::PointCloud<PointType>::Ptr& edge_cloud, ceres::Problem& problem, ceres::LossFunction *loss_function)
    {
        int edgeFeatureCount = 0;
        size_t edgeCloudSize = edge_cloud->points.size();
        for(size_t i = 0; i < edgeCloudSize; ++i)
        {
            PointType currPoints;
            pointAssociaToMap(&edge_cloud->points[i], &currPoints); // 将当前点变换到world坐标系

            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchDis;
            kdtreeEdgeMap->nearestKSearch(currPoints, 5, pointSearchInd, pointSearchDis);  // kd-tree查找
            if(pointSearchDis[4] < 1.0) 
            {
                std::vector<Eigen::Vector3d> nearEdges;
                Eigen::Vector3d center(0, 0, 0);
                for(int j = 0; j < 5; ++j)
                {
                    Eigen::Vector3d cur_tmp(localMapEdge->points[pointSearchInd[j]].x,
                                            localMapEdge->points[pointSearchInd[j]].y,
                                            localMapEdge->points[pointSearchInd[j]].z);
                    center = center + cur_tmp;
                    nearEdges.push_back(cur_tmp);
                }
                center = center / 5.0;  // 计算5个最近邻点的中心点

                Eigen::Matrix3d covMat = Eigen::Matrix3d::Zero();
                for(int j = 0; j < 5; ++j)
                {
                    Eigen::Vector3d VecZeroMean = nearEdges[j] - center;   // 去中心化
                    covMat = covMat + VecZeroMean * VecZeroMean.transpose(); // 协方差矩阵
                }
                // 根据协方差矩阵PCA主成分分析得到5个最近点的主方向，最大的奇异值向量作为主方向
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);
                Eigen::Vector3d unit_direction = saes.eigenvectors().col(2);
                Eigen::Vector3d curr_points(edge_cloud->points[i].x, edge_cloud->points[i].y, edge_cloud->points[i].z);
                if(saes.eigenvalues()[2] > 3 * saes.eigenvalues()[1])  // 最大奇异值足够大说明拟合的线方向符合
                {
                    Eigen::Vector3d point_a, point_b;  // 根据主方向和中心点确定直线端点
                    point_a =  0.1 * unit_direction + center;
                    point_b = -0.1 * unit_direction + center;

                    // 构建残差方程
                    ceres::CostFunction *cost_function = new EdgeCostFunction(curr_points, point_a, point_b);
                    problem.AddResidualBlock(cost_function, loss_function, parameter_opti);
                    edgeFeatureCount++;
                }
            }
        }

        // cout<<"edgeFeatureCount = "<<edgeFeatureCount<<endl;
        if(edgeFeatureCount < 20)
        {
            cout<<"not enough edge feature."<<endl;
        }
    }

    void SurfCostFactor(const pcl::PointCloud<PointType>::Ptr& surf_cloud, ceres::Problem& problem, ceres::LossFunction *loss_function)
    {
        int surfFeatureCount = 0;
        size_t surfCloudSize = surf_cloud->points.size();
        for(size_t i = 0; i < surfCloudSize; ++i)
        {
            PointType currPoints;
            pointAssociaToMap(&surf_cloud->points[i], &currPoints);  //变换坐标

            std::vector<int> pointSearchIds;
            std::vector<float> pointSearchDis;
            kdtreeSurfMap->nearestKSearch(currPoints, 5, pointSearchIds, pointSearchDis);  // kd-tree查找

            Eigen::Matrix<double, 5, 3> matA;
            Eigen::Matrix<double, 5, 1> matB = -1 * Eigen::Matrix<double, 5, 1>::Ones();
            if(pointSearchDis[4] < 1.0)  // 最近邻点满足条件
            {
                for(int j = 0; j < 5; ++j)
                {
                    matA(j, 0) = localMapSurf->points[pointSearchIds[j]].x;
                    matA(j, 1) = localMapSurf->points[pointSearchIds[j]].y;
                    matA(j, 2) = localMapSurf->points[pointSearchIds[j]].z;
                }
                // 拟合平面
                Eigen::Vector3d norm = matA.colPivHouseholderQr().solve(matB);
                double negative_OA_dot_norm = 1.0 / norm.norm();
                norm.normalize();

                bool planeValid = true;
                for(int j = 0; j < 5; ++j)
                {
                    // 将某个点带入平面方程，如果 > 0.2表示平面拟合的不好
                    if(fabs(norm(0) * localMapSurf->points[pointSearchIds[j]].x + 
                            norm(1) * localMapSurf->points[pointSearchIds[j]].y + 
                            norm(2) * localMapSurf->points[pointSearchIds[j]].z + negative_OA_dot_norm) > 0.2)
                    {
                        planeValid = false;
                        break;
                    }
                }

                Eigen::Vector3d curr_point(surf_cloud->points[i].x, surf_cloud->points[i].y, surf_cloud->points[i].z);
                if(planeValid)
                {
                    ceres::CostFunction *cost_function = new SurfCostFunction(curr_point, norm, negative_OA_dot_norm);
                    problem.AddResidualBlock(cost_function, loss_function, parameter_opti);

                    surfFeatureCount++;
                }
            } 
        }

        // cout<<"surfFeatureCount = "<<surfFeatureCount<<endl;
        if(surfFeatureCount < 20)
        {
            cout<<"no enough surf feature."<<endl;
        }

    }


    void optimation_processing(const pcl::PointCloud<PointType>::Ptr& edgeCloud_In, const pcl::PointCloud<PointType>::Ptr& surfCloud_In)
    {

        Eigen::Isometry3d globalOdom_est = globalOdom * (globalOdom_last.inverse() * globalOdom);
        globalOdom_last = globalOdom;
        globalOdom = globalOdom_est;

        q_w_c = Eigen::Quaterniond(globalOdom.rotation());
        t_w_c = globalOdom.translation(); 

        // 特征点下采样
        pcl::PointCloud<PointType>::Ptr voxelEdge_Cloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr voxelSurf_Cloud(new pcl::PointCloud<PointType>());
        voxelEdgeFilter.setInputCloud(edgeCloud_In);
        voxelEdgeFilter.filter(*voxelEdge_Cloud);
        voxelSurfFilter.setInputCloud(surfCloud_In);
        voxelSurfFilter.filter(*voxelSurf_Cloud);

        // local map flann matcher
        if(localMapEdge->points.size() > 10 && localMapSurf->points.size() > 50)
        {
            kdtreeEdgeMap->setInputCloud(localMapEdge);
            kdtreeSurfMap->setInputCloud(localMapSurf);

            // start optimization!
            for(int iter = 0; iter < 2; ++iter)
            {
                // 鲁棒核函数
                ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
                ceres::Problem::Options problem_options;  // 求解器设置
                ceres::Problem problem(problem_options);  // 添加求解器选项

                // 添加参数块
                problem.AddParameterBlock(parameter_opti, 7, new LocalSE3Parameterization());

                // 添加特征约束
                EdgeCostFactor(voxelEdge_Cloud, problem, loss_function);
                SurfCostFactor(voxelSurf_Cloud, problem, loss_function);

                // 优化开始
                ceres::Solver::Options solver_options;
                solver_options.linear_solver_type = ceres::DENSE_QR;
                solver_options.max_num_iterations = 4;
                solver_options.minimizer_progress_to_stdout = false;
                solver_options.check_gradients = false;
                solver_options.gradient_check_relative_precision = 1e-4;
                ceres::Solver::Summary summary;

                ceres::Solve(solver_options, &problem, &summary);
            }
        }
        else{
            cout<<"localMapEdge->points.size() = "<<localMapEdge->points.size()<<" , localMapSurf->points.size() = "<<localMapSurf->points.size()<<endl;
            cout<<"not enough feature points in local map to associate."<<endl;
        }

        globalOdom = Eigen::Isometry3d::Identity();
        globalOdom.linear() = q_w_c.toRotationMatrix();
        globalOdom.translation() = t_w_c;

        createSubMap(voxelEdge_Cloud, voxelSurf_Cloud);  // 特征点云变换到世界坐标系，维护局部地图
    }

    void createSubMap(const pcl::PointCloud<PointType>::Ptr& edge_cloud, const pcl::PointCloud<PointType>::Ptr& surf_cloud)
    {
        cloudRegistered->clear();
        cloudNoRegistered->clear();

        // 未变换到世界坐标系下
        *cloudNoRegistered += *edge_cloud;
        *cloudNoRegistered += *surf_cloud;

        // 1. 特征转换坐标系，并add到local map
        size_t edgeSize = edge_cloud->points.size();
        size_t surfSize = surf_cloud->points.size();
        for(size_t i = 0; i < edgeSize; ++i)
        {
            PointType points;
            pointAssociaToMap(&edge_cloud->points[i], &points);
            localMapEdge->push_back(points);
            cloudRegistered->push_back(points);
        }

        for(size_t i = 0; i < surfSize; ++i)
        {
            PointType points;
            pointAssociaToMap(&surf_cloud->points[i], &points);
            localMapSurf->push_back(points);
            cloudRegistered->push_back(points);
        }

        // 2. 设置局部地图大小，以当前帧位姿为中心，正方体100*100*100扩散
        double x_min = +globalOdom.translation().x() - 100.0;
        double y_min = +globalOdom.translation().y() - 100.0;
        double z_min = +globalOdom.translation().z() - 100.0;
        double x_max = +globalOdom.translation().x() + 100.0;
        double y_max = +globalOdom.translation().y() + 100.0;
        double z_max = +globalOdom.translation().z() + 100.0;

        // 3. box滤波
        cropBoxFilter.setMin(Eigen::Vector4f(x_min, y_min, z_min, 1.0));
        cropBoxFilter.setMax(Eigen::Vector4f(x_max, y_max, z_max, 1.0));
        cropBoxFilter.setNegative(false);

        pcl::PointCloud<PointType>::Ptr cropEdgeCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr cropSurfCloud(new pcl::PointCloud<PointType>());
        cropBoxFilter.setInputCloud(localMapEdge);
        cropBoxFilter.filter(*cropEdgeCloud);
        cropBoxFilter.setInputCloud(localMapSurf);
        cropBoxFilter.filter(*cropSurfCloud);

        // 4. 局部地图下采样
        voxelEdgeFilter.setInputCloud(cropEdgeCloud);
        voxelEdgeFilter.filter(*localMapEdge);
        voxelSurfFilter.setInputCloud(cropSurfCloud);
        voxelSurfFilter.filter(*localMapSurf);

    }


    void pointAssociaToMap(PointType const *const p_in, PointType *const p_out)
    {
        Eigen::Vector3d p_curr(p_in->x, p_in->y, p_in->z);
        Eigen::Vector3d p_world = q_w_c * p_curr + t_w_c;
        p_out->x = p_world.x();
        p_out->y = p_world.y();
        p_out->z = p_world.z();
        p_out->intensity = p_in->intensity;
    }

    void getMapCloud(pcl::PointCloud<PointType>::Ptr& MapRsgistered, pcl::PointCloud<PointType>::Ptr& MapNoRegistered)
    {
        *MapRsgistered = *cloudRegistered;
        *MapNoRegistered = *cloudNoRegistered;
    }

    void getMapCloud(pcl::PointCloud<PointType>::Ptr& MapRsgistered)
    {
        // *MapRsgistered = *cloudRegistered;
        *MapRsgistered = *cloudNoRegistered;
    }


public:

    double edgeMapLeafSize;
    double surfMapLeafSize;

    double parameter_opti[7] = {0, 0, 0, 1, 0, 0, 0};  // q， t
    Eigen::Map<Eigen::Quaterniond> q_w_c = Eigen::Map<Eigen::Quaterniond>(parameter_opti);
    Eigen::Map<Eigen::Vector3d> t_w_c = Eigen::Map<Eigen::Vector3d>(parameter_opti + 4);

    Eigen::Isometry3d globalOdom;
    Eigen::Isometry3d globalOdom_last;

    pcl::CropBox<PointType> cropBoxFilter;
    pcl::PointCloud<PointType>::Ptr cloudRegistered;
    pcl::PointCloud<PointType>::Ptr cloudNoRegistered;

    pcl::PointCloud<PointType>::Ptr localMapEdge;
    pcl::PointCloud<PointType>::Ptr localMapSurf;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeEdgeMap;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfMap;

    pcl::VoxelGrid<PointType> voxelEdgeFilter;
    pcl::VoxelGrid<PointType> voxelSurfFilter;

};