#pragma once

#include "common.h"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/crop_box.h>


class smoothness{

public:

    smoothness(size_t ind_, double value_)
    {
        ind = ind_;
        value = value_;
    }

    double value;
    size_t ind;
};

bool smooth_value(const smoothness& a, const smoothness& b){

    return a.value < b.value;
}


class featureExtraction{

public:
    	featureExtraction()
        {

        }

        void initParam(ros::NodeHandle& nh)
        {
            nh.param<int>("/N_SCAN", N_SCANS, 64);
            nh.param<double>("/lidarMinRange", lidarMinDis, 3.0);
            nh.param<double>("/lidarMaxRange", lidarMaxDis, 100.0);
            nh.param<double>("/edgeThreshold", edgeThreshold, 0.1);
            nh.param<double>("/surfThreshold", surfThreshold, 0.1);
            nh.param<double>("/SurfLeafSize", SurfLeafSize, 0.4);

        }

        void getLaserCloud(const pcl::PointCloud<PointType>::Ptr& cloud_in)
        {
            std::vector<int> indices;
            pcl::removeNaNFromPointCloud(*cloud_in, indices);  // 移除离群点

            for(int i = 0; i < N_SCANS; i++)
            {
                currFullClouds.push_back(pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>()));  // 每条线上激光点
            }

            for (int i = 0; i < (int) cloud_in->points.size(); i++)
            {
                PointType point_tmp = cloud_in->points[i];
                int scanID = 0;
                double distance = DistanceXY(point_tmp);

                if(distance < lidarMinDis || distance > lidarMaxDis)   // 剔除不在范围内的点
                    continue;
                
                double angle = atan(point_tmp.z / distance) * 180 / M_PI;  // 计算垂直方向角度
                
                if (N_SCANS == 16)
                {
                    scanID = int((angle + 15) / 2 + 0.5);
                    if (scanID > (N_SCANS - 1) || scanID < 0)
                    {
                        continue;
                    }
                }
                else if (N_SCANS == 32)
                {
                    scanID = int((angle + 92.0/3.0) * 3.0 / 4.0);
                    if (scanID > (N_SCANS - 1) || scanID < 0)
                    {
                        continue;
                    }
                }
                else if (N_SCANS == 64)
                {   
                    if (angle >= -8.83)
                        scanID = int((2 - angle) * 3.0 + 0.5);
                    else
                        scanID = N_SCANS / 2 + int((-8.83 - angle) * 2.0 + 0.5);

                    if (angle > 2 || angle < -24.33 || scanID > 63 || scanID < 0)
                    {
                        continue;
                    }
                }
                else
                {
                    printf("wrong scan number\n");
                }

                currFullClouds[scanID]->push_back(point_tmp); 
            }
        }

        void featureExtractionFromSector(const pcl::PointCloud<PointType>::Ptr& cloud_in, std::vector<smoothness>& cloudCurvature, pcl::PointCloud<PointType>::Ptr& cloud_Edge, pcl::PointCloud<PointType>::Ptr& cloud_Surf)
        {

            std::sort(cloudCurvature.begin(), cloudCurvature.end(), smooth_value);

            int largestPickedNum = 0;
            std::vector<int> cloudNeighborPicked;  // 标记是否被选为特征点

            for (int i = cloudCurvature.size()-1; i >= 0; i--)  // 从大到小曲率排序点依次遍历
            {
                int ind = cloudCurvature[i].ind; // 激光点在currFullClouds中的id
                if(std::find(cloudNeighborPicked.begin(), cloudNeighborPicked.end(), ind) == cloudNeighborPicked.end()) 
                {
                    if(cloudCurvature[i].value <= edgeThreshold)  // 曲率如果小于阈值，直接跳过
                        break;
                    
                    largestPickedNum++;
                    cloudNeighborPicked.push_back(ind);
                    
                    if (largestPickedNum <= 20) // 每条激光线选取20个点
                    {
                        cloud_Edge->push_back(cloud_in->points[ind]);
                    }else{
                        break;
                    }

                    for(int k = 1; k <= 5; k++)
                    {
                        double diffX = cloud_in->points[ind + k].x - cloud_in->points[ind + k - 1].x;
                        double diffY = cloud_in->points[ind + k].y - cloud_in->points[ind + k - 1].y;
                        double diffZ = cloud_in->points[ind + k].z - cloud_in->points[ind + k - 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }
                        cloudNeighborPicked.push_back(ind + k);
                    }
                    
                    for(int l = -1; l >= -5; l--)
                    {
                        double diffX = cloud_in->points[ind + l].x - cloud_in->points[ind + l + 1].x;
                        double diffY = cloud_in->points[ind + l].y - cloud_in->points[ind + l + 1].y;
                        double diffZ = cloud_in->points[ind + l].z - cloud_in->points[ind + l + 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }
                        cloudNeighborPicked.push_back(ind + l);
                    }

                }
            }
            
            for (int i = 0; i <= (int)cloudCurvature.size()-1; i++)
            {
                int ind = cloudCurvature[i].ind; 
                if( std::find(cloudNeighborPicked.begin(), cloudNeighborPicked.end(), ind) == cloudNeighborPicked.end())
                {
                    cloud_Surf->push_back(cloud_in->points[ind]);
                }
            }
        }

        void featureEdge_Surf(pcl::PointCloud<PointType>::Ptr& cloud_Edge, pcl::PointCloud<PointType>::Ptr& cloud_Surf)
        {
            for(int i = 0; i < N_SCANS; i++)
            {
                if(currFullClouds[i]->points.size() < 131)
                {
                    continue;
                }
                
                cloudCurvature.clear();
                size_t smoothSize = currFullClouds[i]->points.size() - 5;
                
                // 计算平滑度
                for(size_t j = 5; j < smoothSize; j++)
                {
                    double diffX = currFullClouds[i]->points[j - 5].x + currFullClouds[i]->points[j - 4].x + currFullClouds[i]->points[j - 3].x + currFullClouds[i]->points[j - 2].x + currFullClouds[i]->points[j - 1].x 
                                   - 10 * currFullClouds[i]->points[j].x 
                                   + currFullClouds[i]->points[j + 1].x + currFullClouds[i]->points[j + 2].x + currFullClouds[i]->points[j + 3].x + currFullClouds[i]->points[j + 4].x + currFullClouds[i]->points[j + 5].x;
                    double diffY = currFullClouds[i]->points[j - 5].y + currFullClouds[i]->points[j - 4].y + currFullClouds[i]->points[j - 3].y + currFullClouds[i]->points[j - 2].y + currFullClouds[i]->points[j - 1].y 
                                   - 10 * currFullClouds[i]->points[j].y 
                                   + currFullClouds[i]->points[j + 1].y + currFullClouds[i]->points[j + 2].y + currFullClouds[i]->points[j + 3].y + currFullClouds[i]->points[j + 4].y + currFullClouds[i]->points[j + 5].y;
                    double diffZ = currFullClouds[i]->points[j - 5].z + currFullClouds[i]->points[j - 4].z + currFullClouds[i]->points[j - 3].z + currFullClouds[i]->points[j - 2].z + currFullClouds[i]->points[j - 1].z 
                                   - 10 * currFullClouds[i]->points[j].z 
                                   + currFullClouds[i]->points[j + 1].z + currFullClouds[i]->points[j + 2].z + currFullClouds[i]->points[j + 3].z + currFullClouds[i]->points[j + 4].z + currFullClouds[i]->points[j + 5].z;
                    
                    smoothness distance(j, diffX * diffX + diffY * diffY + diffZ * diffZ);
                    cloudCurvature.push_back(distance);
                }

                // 边缘特征点和平面特征点
                cloudSize = smoothSize - 5;
                for(int j = 0; j < 6; j++)
                {
                    int sector_length = (int)(cloudSize / 6);
                    int sector_start = sector_length * j;
                    int sector_end = sector_length * (j+1) - 1;
                    if (j == 5)
                    {
                        sector_end = cloudSize - 1; 
                    }
                    std::vector<smoothness> subCloudCurvature(cloudCurvature.begin() + sector_start, cloudCurvature.begin() + sector_end); 
                    
                    featureExtractionFromSector(currFullClouds[i], subCloudCurvature, cloud_Edge, cloud_Surf);   
                }
            }
        }


		void extractFeature(const pcl::PointCloud<PointType>::Ptr& cloud_in, pcl::PointCloud<PointType>::Ptr& cloud_Edge, pcl::PointCloud<PointType>::Ptr& cloud_Surf)
        {
            currFullClouds.clear();

            // 获取每条激光线上点
            getLaserCloud(cloud_in);

            // 计算激光点平滑度，提取边缘点/平面点特征
            featureEdge_Surf(cloud_Edge, cloud_Surf);
        }

private:

    int N_SCANS;
    double edgeThreshold;
    double surfThreshold;
    double SurfLeafSize;
    double lidarMinDis, lidarMaxDis;

    size_t cloudSize;
    std::vector<smoothness> cloudCurvature; 
    std::vector<pcl::PointCloud<PointType>::Ptr> currFullClouds;

};



