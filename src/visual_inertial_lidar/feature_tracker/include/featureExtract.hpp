#pragma once

#include "common.h"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/extract_indices.h>

#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>


struct smoothness{
    float value;
    size_t ind;
};

bool smooth_value(const smoothness& a, const smoothness& b){

    return a.value < b.value;
}

struct VelodynePointXYZIRT
{
    PCL_ADD_POINT4D;
    PCL_ADD_INTENSITY;
    uint16_t ring;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT (VelodynePointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (uint16_t, ring, ring)
)


using PointXYZIRT = VelodynePointXYZIRT;

class featureExtract{

public:

    featureExtract()
    {

    }

    void allocateMemoy()
    {
        currentCloudIn.reset(new pcl::PointCloud<PointXYZIRT>());
        fullCloud.reset(new pcl::PointCloud<PointType>());
        semanticCloud.reset(new pcl::PointCloud<PointType>());

        surfVoxelFilter.setLeafSize(SurfLeafSize, SurfLeafSize, SurfLeafSize);

        fullCloud->points.resize(N_SCAN * Horizon_SCAN);

        startRingIndex.resize(N_SCAN);
        endRingIndex.resize(N_SCAN);
        pointColInd.resize(N_SCAN * Horizon_SCAN);
        pointRange.resize(N_SCAN * Horizon_SCAN);
        cloudCurvature.resize(N_SCAN * Horizon_SCAN);
        cloudSmoothness.resize(N_SCAN * Horizon_SCAN);
        cloudNeighborPicked.resize(N_SCAN * Horizon_SCAN);
        cloudLabel.resize(N_SCAN * Horizon_SCAN);

        currentCloudIn->clear();
        semanticCloud->clear();

        rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));

    }

    void initParam(ros::NodeHandle& nh)
    {
        nh.param<int>("Horizon_SCAN", Horizon_SCAN, 1800);
        nh.param<int>("N_SCAN", N_SCAN, 64);
        nh.param<int>("downsampleRate", downsampleRate, 1);
        nh.param<double>("lidarMinRange", lidarMinDis, 3.0);
        nh.param<double>("lidarMaxRange", lidarMaxDis, 200.0);
        nh.param<double>("edgeThreshold", edgeThreshold, 1.0);
        nh.param<double>("surfThreshold", surfThreshold, 0.1);
        nh.param<double>("SurfLeafSize", SurfLeafSize, 0.4);
    }


    bool extractFeature(sensor_msgs::PointCloud2& cloud_Msg, pcl::PointCloud<PointType>::Ptr& cloud_Edge, pcl::PointCloud<PointType>::Ptr& cloud_Surf)
    {
        // 指针变量分配内存
        allocateMemoy();

        // ROS消息转换为pcl点云
        pcl::moveFromROSMsg(cloud_Msg, *currentCloudIn);

        // 点云投影
        projectPointCloud();

        // 点云反投影
        inverProjectCloud();

        // 点云曲率
        extractSmoothness();

        // 标记被遮挡的点
        markBadPoints();

        // 提取边缘点/面
        featureEdge_Surf(cloud_Edge, cloud_Surf);

    }

    // 边缘点/平面点提取
    void featureEdge_Surf(pcl::PointCloud<PointType>::Ptr& cloud_Edge, pcl::PointCloud<PointType>::Ptr& cloud_Surf)
    {
        pcl::PointCloud<PointType>::Ptr surfCloudScan(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surfCloudScanDS(new pcl::PointCloud<PointType>());

        for(int i = 0; i < N_SCAN; ++i)
        {
            surfCloudScan->clear();

            for(int j = 0; j < 6; j++)
            {
                // 起始点和终止点
                int sp = (startRingIndex[i] * (6-j) + endRingIndex[i] * j) / 6;
                int ep = (startRingIndex[i] * (5-j) + endRingIndex[i] * (j+1)) / 6 - 1;

                if(sp >= ep)
                    continue;
                
                // 按照点平滑度从小到大排序
                std::sort(cloudSmoothness.begin()+sp, cloudSmoothness.begin()+ep, smooth_value);

                // 1. 边缘点标记
                int largestPickedNum = 0;
                for(int k = ep; k >= sp; k--)
                {
                    int ind = cloudSmoothness[k].ind;
                    // 满足未被遮挡和平滑度小于阈值点设置为边缘点
                    if(cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] > edgeThreshold)
                    {
                        largestPickedNum++;
                        if(largestPickedNum <= 20)  // 每条激光线上最多提取20个边缘点
                        {
                            cloudLabel[ind] = 1;
                            cloud_Edge->push_back(semanticCloud->points[ind]);
                        }
                        else
                        {
                            break;
                        }

                        cloudNeighborPicked[ind] = 1; //标记点已被使用
                        // 标记当前点的后5个点
                        for(int l = 1; l <= 5; l++)
                        {
                            int columnDiff = std::abs(int(pointColInd[ind + l] - pointColInd[ind + l - 1]));
                            if(columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;  // 被计算曲率的相同点也被跳过，每10个为一组
                        }
                        // 标记当前点的前5个点
                        for(int l = -1; l >= -5; l--)
                        {
                            int columnDiff = std::abs(int(pointColInd[ind + l] - pointColInd[ind + l + 1]));
                            if(columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;     
                        }
                    }
                }

                // 2.平面点标记
                for(int k = sp; k < ep; k++)
                {
                    int ind = cloudSmoothness[k].ind;
                    if(cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] < surfThreshold)
                    {
                        cloudLabel[ind] = -1;
                        cloudNeighborPicked[ind] = 1;

                        for(int l = 1; l <= 5; l++)
                        {
                            int columDiff =  std::abs(int(pointColInd[ind + l] - pointColInd[ind + l - 1]));
                            if(columDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }

                        for(int l = -1; l >= -5; l--)
                        {
                            int columDiff = std::abs(int(pointColInd[ind + l] - pointColInd[ind + l + 1]));
                            if(columDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }

                // 3.添加每条线束上的平面点, 除了边缘点和遮挡点之外，都被标记为平面点
                for(int k = sp; k <= ep; k++)
                {
                    if(cloudLabel[k] <= 0) 
                        surfCloudScan->push_back(semanticCloud->points[k]);
                }
            }

            // // 对每条线束上平面点降采样
            // surfCloudScanDS->clear();
            // surfVoxelFilter.setInputCloud(surfCloudScan);
            // surfVoxelFilter.filter(*surfCloudScanDS);
            // *cloud_Surf += *surfCloudScanDS; 

            *cloud_Surf += *surfCloudScan; 
        }

    }

    // 标记遮挡点
    void markBadPoints()
    {
        size_t cloudSize = semanticCloud->points.size();
        for(size_t i = 5; i < cloudSize-6; ++i)
        {
            // 标记被遮挡点
            float depth1 = pointRange[i];
            float depth2 = pointRange[i+1];
            int columnDiff = std::abs(int(pointColInd[i+1] - pointColInd[i]));

            if(columnDiff < 10)
            {
               if(depth1 - depth2 > 0.3)
               {
                   cloudNeighborPicked[i-5] = 1;
                   cloudNeighborPicked[i-4] = 1;
                   cloudNeighborPicked[i-3] = 1;
                   cloudNeighborPicked[i-2] = 1;
                   cloudNeighborPicked[i-1] = 1;
                   cloudNeighborPicked[i]   = 1; 
               }
               else if(depth2 - depth1 > 0.3)
               {
                   cloudNeighborPicked[i+6] = 1;
                   cloudNeighborPicked[i+5] = 1;
                   cloudNeighborPicked[i+4] = 1;
                   cloudNeighborPicked[i+3] = 1;
                   cloudNeighborPicked[i+2] = 1;
                   cloudNeighborPicked[i+1] = 1; 
               }
            }

            float diff1 = std::abs(float(pointRange[i-1] - pointRange[i]));
            float diff2 = std::abs(float(pointRange[i+1] - pointRange[i]));

            if(diff1 > 0.02  * pointRange[i] && diff2 > 0.02 * pointRange[i])
                cloudNeighborPicked[i] = 1;
        }
    }

    // 点云平滑度计算
    void extractSmoothness()
    {
        size_t cloudSize = semanticCloud->points.size();
        for(size_t i = 5; i < cloudSize-5; ++i)
        {
            float diffRange = pointRange[i-5] +  pointRange[i-4] + pointRange[i-3] + pointRange[i-2] + pointRange[i-1]
                            + pointRange[i+5] +  pointRange[i+4] + pointRange[i+3] + pointRange[i+2] + pointRange[i+1]
                            - pointRange[i] * 10;
            
            cloudCurvature[i] = diffRange * diffRange;

            cloudNeighborPicked[i] = 0;
            cloudLabel[i] = 0;

            // 点曲率用于排序
            cloudSmoothness[i].ind = i;
            cloudSmoothness[i].value = cloudCurvature[i];
        }
    }

    // 点云反投影
    void inverProjectCloud()
    {
        int count = 0;
        for(int i = 0; i < N_SCAN; ++i)
        {
            // 起始行id
            startRingIndex[i] = count - 1 + 5;

            for(int j = 0; j < Horizon_SCAN; ++j)
            {
                if(rangeMat.at<float>(i, j) != FLT_MAX)
                {
                    pointColInd[count] = j;
                    pointRange[count] = rangeMat.at<float>(i, j);  
                    semanticCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                    ++count;
                }
            }

            endRingIndex[i] = count - 1 -5;
        }

    }

    // 点云投影
    void projectPointCloud()
    {
        size_t cloudSize = currentCloudIn->points.size();
        // 区域范围点云投影
        for(size_t i = 0; i < cloudSize; i++)
        {
            PointType thisPoint;
            thisPoint.x = currentCloudIn->points[i].x;
            thisPoint.y = currentCloudIn->points[i].y;
            thisPoint.z = currentCloudIn->points[i].z;
            thisPoint.intensity  = currentCloudIn->points[i].intensity;

            // 计算当前激光点到原点距离
            float range = pointDistance(thisPoint); 
            float rangeDistance = DistanceXY(thisPoint); 

            // 距离范围之外点云剔除
            if(rangeDistance < lidarMinDis || rangeDistance > lidarMaxDis)
                continue;

            // 激光雷达线束，投影图行数
            int rowIdn = currentCloudIn->points[i].ring;
            if(rowIdn < 0 || rowIdn >= N_SCAN)
                continue;
            
            // 用于降采样，设置为2表示值选用偶数线束
            if(rowIdn % downsampleRate != 0)
                continue;
                  
            float horizonAngle = rad2deg( atan2(thisPoint.x, thisPoint.y) ); // 转换为角度
            
            static float ang_res_x = 360.0 / float(Horizon_SCAN);
            // 投影图的列数
            int columnIdn = -round((horizonAngle - 90.0) / ang_res_x) + Horizon_SCAN/2; // 从270°为起点，逆时针旋转
            if(columnIdn >= Horizon_SCAN)
                columnIdn -= Horizon_SCAN;


            if(columnIdn < 0 || columnIdn >= Horizon_SCAN)
                continue;

            if(rangeMat.at<float>(rowIdn, columnIdn) != FLT_MAX)
                continue;
            
                   
            // 根据时间戳矫正点云畸变（待补充）
            
            rangeMat.at<float>(rowIdn, columnIdn) = range;  // 投影为图像，单通道
            int index = columnIdn  + rowIdn * Horizon_SCAN; // 点云id
            fullCloud->points[index] = thisPoint;
        }
    }



private:

    int N_SCAN;
    int Horizon_SCAN;
    int downsampleRate;
    cv::Mat rangeMat;
    double edgeThreshold;
    double surfThreshold;
    double SurfLeafSize;

    double lidarMinDis, lidarMaxDis;
    std::vector<int> startRingIndex;
    std::vector<int> endRingIndex;
    std::vector<int> pointColInd;
    std::vector<float> pointRange;
    std::vector<float> cloudCurvature;
    std::vector<int> cloudLabel;
    std::vector<int> cloudNeighborPicked;
    std::vector<smoothness> cloudSmoothness;

    pcl::PointCloud<PointXYZIRT>::Ptr currentCloudIn;
    pcl::PointCloud<PointType>::Ptr fullCloud;
    pcl::PointCloud<PointType>::Ptr semanticCloud;

    pcl::VoxelGrid<PointType> surfVoxelFilter;

    

};