#include "include/Scancontext/Scancontext.h"

#include <queue>
#include <thread>
#include <mutex>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/crop_box.h>
#include <pcl/registration/icp.h>

#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Rot2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/ISAM2.h>

#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

using namespace std;
using namespace gtsam;

ros::Publisher pubPathGraph, pubOdomGraph, pubMapGraph, pubMapCloud;

queue<sensor_msgs::PointCloud2ConstPtr> cloudQueue;
queue<nav_msgs::OdometryConstPtr> odomQueue;

std::mutex buf_mutex;
std::mutex kF_mutex;

double odometryTime;

// 关键帧变量
int recentIdxUpdated = 0;
std::vector<keyFrameCloud> KeyFrames;
std::vector<pcl::PointCloud<PointType>::Ptr> KeyFrameClouds;
bool isNowKeyFrame = false;  // 关键帧标志位
double keyFrameTransTh;      // 平移阈值
double keyFrameRotateTh;     // 旋转阈值
double translateAcc = 1000000.0;
double rotationAcc = 100000.0;
Pose6D prevOdomPose {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
Pose6D currOdomPose {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

// ScanContex—闭环检测对象
SCManager scManager;
double scDistThres;
double scMaxRadius;
bool isValidSCloop = false;

// 闭环检测
queue<std::pair<int, int> > scLoopICPBuf;

// gtsam optimization 优化
gtsam::ISAM2 *isam;
std::mutex poseGraph_mutex;
bool gtsamGraphMade = false;
gtsam::NonlinearFactorGraph gtsamGraph;
gtsam::Values initialEstimate;
gtsam::Values isamCurrentEstimate;

// 噪声相关参数
noiseModel::Diagonal::shared_ptr priorNoise;
noiseModel::Diagonal::shared_ptr odomNoise;
noiseModel::Base::shared_ptr robustLoopNoise;

// Map相关参数
double mapLeafSize;
pcl::VoxelGrid<PointType> downMapFilter;
pcl::PointCloud<PointType>::Ptr GraphMapCloud(new pcl::PointCloud<PointType>());

std::string poseSaveTum;
double firstTimes = 0.0;

void saveTUMTrajOdometry()
{
    // ref from gtsam's original code "dataset.cpp"
    std::fstream stream(poseSaveTum + "fs_loam_loop.txt", std::fstream::out);
    stream.setf(ios::fixed, ios::floatfield);
    for(const auto& keyFrame : KeyFrames)
    {
        Pose6D pose6d = keyFrame.KeyFramePoses;
        tf::Quaternion quat = tf::createQuaternionFromRPY(pose6d.roll, pose6d.pitch, pose6d.yaw);
        stream.precision(9);
        // stream << keyFrame.KeyFrameTimes - firstTimes <<" ";
        stream << keyFrame.KeyFrameTimes <<" ";
        stream.precision(5);
        stream << pose6d.x<<" "
               << pose6d.y<<" "
               << pose6d.z<<" "
               << quat.getX()<<" "
               << quat.getY()<<" "
               << quat.getZ()<<" "
               << quat.getW()<<endl;
    }
    stream.close();
}

void callbackVelodyne(const sensor_msgs::PointCloud2ConstPtr& velodyne_msgs)
{
    buf_mutex.lock();
    cloudQueue.push(velodyne_msgs);
    buf_mutex.unlock();
}

void callbackOdometry(const nav_msgs::OdometryConstPtr& odom_msgs)
{
    buf_mutex.lock();
    odomQueue.push(odom_msgs);
    buf_mutex.unlock();
}

void initNoises( void )
{
    gtsam::Vector priorNoiseVector6(6);
    priorNoiseVector6 << 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12;
    priorNoise = noiseModel::Diagonal::Variances(priorNoiseVector6);

    gtsam::Vector odomNoiseVector6(6);
    odomNoiseVector6 << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4;
    odomNoise = noiseModel::Diagonal::Variances(odomNoiseVector6);

    double loopNoiseScore = 0.5; // constant is ok...
    gtsam::Vector robustNoiseVector6(6); // gtsam::Pose3 factor has 6 elements (6D)
    robustNoiseVector6 << loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore;
    robustLoopNoise = gtsam::noiseModel::Robust::Create(
                    gtsam::noiseModel::mEstimator::Cauchy::Create(1), // optional: replacing Cauchy by DCS or GemanMcClure is okay but Cauchy is empirically good.
                    gtsam::noiseModel::Diagonal::Variances(robustNoiseVector6) );
} 

Pose6D getOdom(nav_msgs::Odometry::ConstPtr _odom)
{
    double tx = _odom->pose.pose.position.x;
    double ty = _odom->pose.pose.position.y;
    double tz = _odom->pose.pose.position.z;

    double roll, pitch, yaw;
    geometry_msgs::Quaternion quat = _odom->pose.pose.orientation;
    tf::Matrix3x3(tf::Quaternion(quat.x, quat.y, quat.z, quat.w)).getRPY(roll, pitch, yaw);

    return Pose6D{tx, ty, tz, roll, pitch, yaw}; 
} 

Pose6D diffTransformation(const Pose6D& _p1, const Pose6D& _p2)
{
    Eigen::Affine3f SE3_p1 = pcl::getTransformation(_p1.x, _p1.y, _p1.z, _p1.roll, _p1.pitch, _p1.yaw);
    Eigen::Affine3f SE3_p2 = pcl::getTransformation(_p2.x, _p2.y, _p2.z, _p2.roll, _p2.pitch, _p2.yaw);
    Eigen::Matrix4f SE3_delta0 = SE3_p1.matrix().inverse() * SE3_p2.matrix();
    Eigen::Affine3f SE3_delta; SE3_delta.matrix() = SE3_delta0;
    float dx, dy, dz, droll, dpitch, dyaw;
    pcl::getTranslationAndEulerAngles (SE3_delta, dx, dy, dz, droll, dpitch, dyaw);

    return Pose6D{double(abs(dx)), double(abs(dy)), double(abs(dz)), double(abs(droll)), double(abs(dpitch)), double(abs(dyaw))};
} 

gtsam::Pose3 Pose6DtoGTSAMPose3(const Pose6D& p)
{
    return gtsam::Pose3( gtsam::Rot3::RzRyRx(p.roll, p.pitch, p.yaw), gtsam::Point3(p.x, p.y, p.z) );
}

pcl::PointCloud<PointType>::Ptr local2global(const pcl::PointCloud<PointType>::Ptr &cloudIn, const Pose6D& tf)
{
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

    int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);

    Eigen::Affine3f transCur = pcl::getTransformation(tf.x, tf.y, tf.z, tf.roll, tf.pitch, tf.yaw);
    
    // int numberOfCores = 8;
    // #pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < cloudSize; ++i)
    {
        PointType pointFrom = cloudIn->points[i];
        cloudOut->points[i].x = transCur(0,0) * pointFrom.x + transCur(0,1) * pointFrom.y + transCur(0,2) * pointFrom.z + transCur(0,3);
        cloudOut->points[i].y = transCur(1,0) * pointFrom.x + transCur(1,1) * pointFrom.y + transCur(1,2) * pointFrom.z + transCur(1,3);
        cloudOut->points[i].z = transCur(2,0) * pointFrom.x + transCur(2,1) * pointFrom.y + transCur(2,2) * pointFrom.z + transCur(2,3);
        cloudOut->points[i].intensity = pointFrom.intensity;
    }

    return cloudOut;
}

void loopFindNearKeyframeCLoud(pcl::PointCloud<PointType>::Ptr& nearKeyframe, const int& key, const int& submap_size, const int& root_idx)
{
    nearKeyframe->clear();
    for(int i = -submap_size; i <= submap_size; i++)
    {
        // int keyNear = root_idx + i;
        int keyNear = key + i;
        if(keyNear < 0 || keyNear >= int(KeyFrameClouds.size()))
            continue;
        
        kF_mutex.lock();
        // *nearKeyframe += *local2global(KeyFrameClouds[keyNear], KeyFrames[keyNear].KeyFramePosesUpdated);
        *nearKeyframe += *local2global(KeyFrameClouds[keyNear], KeyFrames[root_idx].KeyFramePosesUpdated);
        kF_mutex.unlock();
    }

    if(nearKeyframe->empty())
        return;
    
    // 下采样
    pcl::PointCloud<PointType>::Ptr cloud_tmp(new pcl::PointCloud<PointType>());
    downMapFilter.setInputCloud(nearKeyframe);
    downMapFilter.filter(*cloud_tmp);

    *nearKeyframe = *cloud_tmp;
}

void updatePoses(void)
{
    kF_mutex.lock(); 
    for (int node_idx=0; node_idx < int(isamCurrentEstimate.size()); node_idx++)
    {
        Pose6D& p =KeyFrames[node_idx].KeyFramePosesUpdated;
        p.x = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).translation().x();
        p.y = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).translation().y();
        p.z = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).translation().z();
        p.roll = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).rotation().roll();
        p.pitch = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).rotation().pitch();
        p.yaw = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).rotation().yaw();
    }

    recentIdxUpdated = int(KeyFrames.size()) - 1;
    kF_mutex.unlock();
} 

void loopPath()
{
    float pathRate = 10.0; // path频率
    ros::Rate rate(pathRate);
    while(ros::ok())
    {
        rate.sleep();
        if( recentIdxUpdated > 1 )
        {
            // pub odom and path 
            nav_msgs::Odometry odomGraph;
            nav_msgs::Path pathGraph;
            kF_mutex.lock(); 

            std::fstream stream(poseSaveTum + "fs_loam_loop.txt", std::fstream::out);
            stream.setf(ios::fixed, ios::floatfield);

            for (int node_idx=0; node_idx < recentIdxUpdated; node_idx++) // -1 is just delayed visualization (because sometimes mutexed while adding(push_back) a new one)
            {
                const Pose6D& pose_est = KeyFrames.at(node_idx).KeyFramePosesUpdated; // upodated poses

                odomGraph.header.frame_id = "world";
                odomGraph.child_frame_id = "/aft_Graph";
                odomGraph.header.stamp = ros::Time().fromSec(KeyFrames.at(node_idx).KeyFrameTimes);
                odomGraph.pose.pose.position.x = pose_est.x;
                odomGraph.pose.pose.position.y = pose_est.y;
                odomGraph.pose.pose.position.z = pose_est.z;
                odomGraph.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(pose_est.roll, pose_est.pitch, pose_est.yaw);

                geometry_msgs::PoseStamped poseStampGraph;
                poseStampGraph.header = odomGraph.header;
                poseStampGraph.pose = odomGraph.pose.pose;

                pathGraph.header.frame_id = "world";
                pathGraph.header.stamp = odomGraph.header.stamp;
                pathGraph.poses.push_back(poseStampGraph);

                // ref from gtsam's original code "dataset.cpp"
                tf::Quaternion quat = tf::createQuaternionFromRPY(pose_est.roll, pose_est.pitch, pose_est.yaw);
                stream.precision(9);
                // stream << odomGraph.header.stamp.toSec() - firstTimes <<" ";
                stream << odomGraph.header.stamp.toSec() <<" ";
                stream.precision(5);
                stream << pose_est.x<<" "
                    << pose_est.y<<" "
                    << pose_est.z<<" "
                    << quat.getX()<<" "
                    << quat.getY()<<" "
                    << quat.getZ()<<" "
                    << quat.getW()<<endl;
            }
            stream.close();

            kF_mutex.unlock(); 
            pubOdomGraph.publish(odomGraph); // last pose 
            pubPathGraph.publish(pathGraph); // poses 

            static tf::TransformBroadcaster br;
            tf::Transform transform;
            tf::Quaternion q;
            transform.setOrigin(tf::Vector3(odomGraph.pose.pose.position.x, odomGraph.pose.pose.position.y, odomGraph.pose.pose.position.z));
            q.setW(odomGraph.pose.pose.orientation.w);
            q.setX(odomGraph.pose.pose.orientation.x);
            q.setY(odomGraph.pose.pose.orientation.y);
            q.setZ(odomGraph.pose.pose.orientation.z);
            transform.setRotation(q);
            br.sendTransform(tf::StampedTransform(transform, odomGraph.header.stamp, "world", "/aft_Graph"));  
        }
    }
}

void publishGlobalMap()
{
    if( recentIdxUpdated > 1 )
    {
        int SKIP_FRAMES = 1; // sparse map visulalization to save computations 
        int counter = 0;

        kF_mutex.lock(); 
        for (int node_idx=0; node_idx < recentIdxUpdated; node_idx++) {
            if(counter % SKIP_FRAMES == 0) {
                *GraphMapCloud += *local2global(KeyFrameClouds[node_idx], KeyFrames[node_idx].KeyFramePosesUpdated);
            }
            counter++;
        }
        kF_mutex.unlock(); 

        downMapFilter.setInputCloud(GraphMapCloud);
        downMapFilter.filter(*GraphMapCloud);

        sensor_msgs::PointCloud2 GraphMapCloudMsg;
        pcl::toROSMsg(*GraphMapCloud, GraphMapCloudMsg);
        GraphMapCloudMsg.header.frame_id = "world";
        pubMapGraph.publish(GraphMapCloudMsg);

        GraphMapCloud->clear();
    }
}

void loopMap()
{
    float mapRate = 0.2;
    ros::Rate rate(mapRate);
    while(ros::ok())
    {
        rate.sleep();
        publishGlobalMap();
    }
}

void isamUpdate()
{
    float updateRate = 1.0;
    ros::Rate rate(updateRate);
    while(ros::ok())
    {
        rate.sleep();
        if( gtsamGraphMade )
        {
            poseGraph_mutex.lock();

            // 更新因子和测量值
            isam->update(gtsamGraph, initialEstimate);
            isam->update();

            gtsamGraph.resize(0);
            initialEstimate.clear();
            isamCurrentEstimate = isam->calculateEstimate();
            updatePoses();

            poseGraph_mutex.unlock();

            // saveTUMTrajOdometry();
        }
    }
}

void icpCalculation()
{
    while(1)
    {
        while( !scLoopICPBuf.empty() )
        {
            if( scLoopICPBuf.size() > 30 ) 
                cout<<"too many loop candidates to waitting, please less frequency (adjust loopClosureFrequency)."<<endl;
            
            buf_mutex.lock();
            std::pair<int, int> loopIdx_pair = scLoopICPBuf.front();
            scLoopICPBuf.pop();
            buf_mutex.unlock();

            const int prevNode_idx = loopIdx_pair.first;
            const int currNode_idx = loopIdx_pair.second;

            // icp求解相对位姿
            int historyKeyframesSearchNum = 25;
            pcl::PointCloud<PointType>::Ptr currentKeyframeCloud(new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr targetKeyframeCloud(new pcl::PointCloud<PointType>());
            loopFindNearKeyframeCLoud(currentKeyframeCloud, currNode_idx, 0, prevNode_idx);   // 查找最近25帧的闭环帧，并闭环到相同坐标系
            loopFindNearKeyframeCLoud(targetKeyframeCloud, prevNode_idx, historyKeyframesSearchNum, prevNode_idx);

            // icp配准参数
            pcl::IterativeClosestPoint<PointType, PointType> icp;
            icp.setMaxCorrespondenceDistance(100);
            icp.setMaximumIterations(100);
            icp.setTransformationEpsilon(1e-6);
            icp.setEuclideanFitnessEpsilon(1e-6);
            icp.setRANSACIterations(0);

            // icp开始配准求解
            icp.setInputSource(currentKeyframeCloud);
            icp.setInputTarget(targetKeyframeCloud);
            pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
            icp.align(*unused_result);

            float loopFitnessScoreThreshold = 0.3;
            if(icp.hasConverged() == false || icp.getFitnessScore() > loopFitnessScoreThreshold)
            {
                cout<<"[SC loop] fitness is failed, and fitness score is "<<icp.getFitnessScore()<<" , it is > "<<loopFitnessScoreThreshold<<endl;
                isValidSCloop = false;
            }
            else{
                cout<<"[SC loop] fitness is successed, and fitness score is "<<icp.getFitnessScore()<<endl;
                isValidSCloop = true;
            }

            if(isValidSCloop == true)
            {
                // 直接转换为相对位姿
                float x, y, z, roll, yaw, pitch;
                Eigen::Affine3f correctionLidarFrame;
                correctionLidarFrame = icp.getFinalTransformation();
                pcl::getTranslationAndEulerAngles (correctionLidarFrame, x, y, z, roll, pitch, yaw);
                gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
                gtsam::Pose3 poseTo   = Pose3(Rot3::RzRyRx(0.0, 0.0, 0.0), Point3(0.0, 0.0, 0.0));

                poseGraph_mutex.lock();
                gtsamGraph.add(gtsam::BetweenFactor<gtsam::Pose3>(prevNode_idx, currNode_idx, poseFrom.between(poseTo), robustLoopNoise));
                poseGraph_mutex.unlock();
            }
        }

        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}

void poseGraphOptimiza()
{

    while(1)
    {
        if( !cloudQueue.empty() && !odomQueue.empty() )
        {
            // 同步点云和里程计数据
            buf_mutex.lock();
            if( !odomQueue.empty() && ( cloudQueue.front()->header.stamp.toSec() < odomQueue.front()->header.stamp.toSec() ) )
            {
                cloudQueue.pop();
                cout<<"pose gaph time cloud pop."<<endl;
                buf_mutex.unlock();

                continue;
            }

            if( !cloudQueue.empty() && ( odomQueue.front()->header.stamp.toSec() < cloudQueue.front()->header.stamp.toSec()))
            {
                odomQueue.pop();
                cout<<"pose graph time odometry pop."<<endl;
                buf_mutex.unlock();

                continue;
            }

            // 同步数据完成，读取数据
            nav_msgs::OdometryConstPtr laserOdometry = odomQueue.front();
            odomQueue.pop();

            pcl::PointCloud<PointType>::Ptr thisKeyFrame(new pcl::PointCloud<PointType>());
            pcl::fromROSMsg(*cloudQueue.front(), *thisKeyFrame);
            cloudQueue.pop();

            buf_mutex.unlock();

            odometryTime = laserOdometry->header.stamp.toSec();
            Pose6D currPose = getOdom(laserOdometry);

            if(firstTimes <= 0.01)
                firstTimes = odometryTime;

            /* pose odomtry */
            pcl::PointCloud<PointType>::Ptr transformCloud(new pcl::PointCloud<PointType>());
            Eigen::Matrix4d tform;

            Eigen::Quaterniond quat;
            Eigen::Vector3d trans;
            quat.x() = laserOdometry->pose.pose.orientation.x;
            quat.y() = laserOdometry->pose.pose.orientation.y;
            quat.z() = laserOdometry->pose.pose.orientation.z;
            quat.w() = laserOdometry->pose.pose.orientation.w;
            trans.x() = laserOdometry->pose.pose.position.x;
            trans.y() = laserOdometry->pose.pose.position.y;
            trans.z() = laserOdometry->pose.pose.position.z;

            tform.block(0,0,3,3) = quat.toRotationMatrix();
            tform(0, 3) = trans[0];
            tform(1, 3) = trans[1];
            tform(2, 3) = trans[2];
            tform(3, 0) = 0; tform(3, 1) = 0; tform(3, 2) = 0; tform(3, 3) = 1;
        
            pcl::transformPointCloud(*thisKeyFrame, *transformCloud, tform);  // 转换拼接点云

            // 发布处理后的点云消息
            sensor_msgs::PointCloud2 laserCloudOutMsg;
            pcl::toROSMsg(*transformCloud, laserCloudOutMsg);
            laserCloudOutMsg.header.stamp = laserOdometry->header.stamp;
            laserCloudOutMsg.header.frame_id = "world";
            pubMapCloud.publish(laserCloudOutMsg);

            // 根据相对运动距离和旋转筛选关键帧
            prevOdomPose = currOdomPose;
            currOdomPose = currPose;
            Pose6D  relativePose = diffTransformation(prevOdomPose, currOdomPose);

            double delta_translation = poseDistance(relativePose);  // 相对位姿的绝对距离
            translateAcc += delta_translation;
            rotationAcc += (relativePose.roll + relativePose.pitch + relativePose.yaw);

            if( translateAcc > keyFrameTransTh || rotationAcc > keyFrameRotateTh)
            {
                isNowKeyFrame = true;
                translateAcc = 0.0;
                rotationAcc  = 0.0;
            }
            else{
                isNowKeyFrame = false;
            }

            if( !isNowKeyFrame )  // 非关键帧直接跳过
                continue;
            
            // 当前点云是关键帧， 保存数据添加关联节点
            pcl::PointCloud<PointType>::Ptr thisKeyFrameDS(new pcl::PointCloud<PointType>());
            *thisKeyFrameDS = *thisKeyFrame;  // 由于前面数据已进行下采样，便不再降采样

            // 关键帧插入
            kF_mutex.lock();
            keyFrameCloud tempKeyFrame;
            tempKeyFrame.KeyFramePoses  = currPose;
            tempKeyFrame.KeyFramePosesUpdated = currPose;
            tempKeyFrame.KeyFrameTimes  = odometryTime;
            KeyFrames.push_back(tempKeyFrame);
            KeyFrameClouds.push_back(thisKeyFrameDS);

            scManager.makeAndSaveScancontextAndKeys(*thisKeyFrameDS);  // 关键帧转换为contex
            kF_mutex.unlock();

            // 添加pose节点, 包括第一帧固定节点和相邻关键帧的位姿节点
            const int prevNode_idx = KeyFrames.size() - 2;   
            const int currNode_idx = KeyFrames.size() - 1;   
            if( ! gtsamGraphMade )
            {
                const int initNode_idx = 0;
                gtsam::Pose3 poseOrigin = Pose6DtoGTSAMPose3( KeyFrames.at(initNode_idx).KeyFramePoses );

                poseGraph_mutex.lock();
                {
                    // 先验因子
                    gtsamGraph.add(gtsam::PriorFactor<gtsam::Pose3>(initNode_idx, poseOrigin, priorNoise));
                    initialEstimate.insert(initNode_idx, poseOrigin);
                }
                poseGraph_mutex.unlock();

                gtsamGraphMade = true;

                cout<<"add prior node: "<<initNode_idx<<" optimization."<<endl;
            }else{
                gtsam::Pose3 poseFrom = Pose6DtoGTSAMPose3( KeyFrames.at(prevNode_idx).KeyFramePoses );
                gtsam::Pose3 poseTo   = Pose6DtoGTSAMPose3( KeyFrames.at(currNode_idx).KeyFramePoses );

                poseGraph_mutex.lock();
                {
                    gtsam::Pose3 relPose = poseFrom.between(poseTo);  // 关键帧间相对位姿
                    gtsamGraph.add( gtsam::BetweenFactor<gtsam::Pose3>(prevNode_idx, currNode_idx, relPose, odomNoise) ); // 添加相对位姿约束
                    initialEstimate.insert(currNode_idx, poseTo);
                }
                poseGraph_mutex.unlock();

                if(currNode_idx % 100 == 0)
                    cout<<"add relative pose node: "<<currNode_idx<<" optimization."<<endl;
            }
            
        }

        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}

void performSCLoopDetection()
{
    if( int(KeyFrames.size()) < scManager.NUM_EXCLUDE_RECENT )
        return ;
    
    std::pair<int, float> detectResult = scManager.detectLoopClosureID();  // 闭环帧id以及yaw差距
    int SCclosestHistoryFrameID = detectResult.first;
    if(SCclosestHistoryFrameID != -1)
    {
        const int prevNode_idx = SCclosestHistoryFrameID;
        const int currNode_idx = KeyFrames.size() - 1; 
        cout<<"detection close keyframes - between "<<prevNode_idx<<" and "<<currNode_idx<<" "<<endl;

        buf_mutex.lock();
        scLoopICPBuf.push(std::pair<int, int>(prevNode_idx, currNode_idx));  // 将闭环对应边因子存储
        buf_mutex.unlock();
    }
}

void loopDetection()
{
    float loopFrequency = 1.0;  // 闭环频率
    ros::Rate rate(loopFrequency);
    while(ros::ok())
    {
        rate.sleep();
        // 执行闭环
        performSCLoopDetection();
    }
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "sensor_fusion_poseGraphOptimization");
    ros::NodeHandle nh;

    // 关键帧相关参数
	nh.param<double>("/keyframe_trans_th", keyFrameTransTh, 2.0); // pose assignment every k m move 
	nh.param<double>("/keyframe_rotate_th", keyFrameRotateTh, 10.0); // pose assignment every k deg rot 
    keyFrameRotateTh = deg2rad(keyFrameRotateTh);  // 转换为弧度

    // ScanContex相关参数
    nh.param<double>("/sc_dist_th", scDistThres, 0.2);
    nh.param<double>("/sc_max_rad", scMaxRadius, 80.0);
    scManager.setSCdistThres(scDistThres);
    scManager.setMaximumRadius(scMaxRadius);

    // Map相关参数
    nh.param<double>("/MapLeafSize", mapLeafSize, 0.4);
    downMapFilter.setLeafSize(mapLeafSize, mapLeafSize, mapLeafSize);

    // gtsam优化参数初始化
    ISAM2Params parameters;
    parameters.relinearizeSkip = 1;
    parameters.relinearizeThreshold = 0.01;
    isam = new ISAM2(parameters);  // new初始化对象

    // 噪声相关参数
    initNoises();

    nh.param<string>("/poseSaveTum", poseSaveTum, "/home/lenovo/output/TaoZi/");

    ros::Subscriber subLidarCloudFull = nh.subscribe<sensor_msgs::PointCloud2>("/GlobalMap", 100, callbackVelodyne);
    ros::Subscriber subLidarOdometry  = nh.subscribe<nav_msgs::Odometry>("/sensor_fusion_eatimator/odometry", 100, callbackOdometry); // /sensor_fusion_eatimator/odometry
    
    
    pubOdomGraph = nh.advertise<nav_msgs::Odometry>("/Odometry_graph", 100);
    pubPathGraph = nh.advertise<nav_msgs::Path>("/Path_graph", 100);
    pubMapGraph  = nh.advertise<sensor_msgs::PointCloud2>("/Map_graph", 100);
    pubMapCloud  = nh.advertise<sensor_msgs::PointCloud2>("/Map_Cloud", 100);

    std::thread posgGraphProcessing {poseGraphOptimiza};   // 位姿图优化
    std::thread loopDetectionProcessing {loopDetection};   // 闭环检测
    std::thread icpCalculationProcessing {icpCalculation}; // 闭环优化icp求解相对位姿
    std::thread isamUpdateProcessing {isamUpdate};         // isam开始更新因子优化因子图

    // std::thread loopMapProcessing  {loopMap};              // 低频地图
    std::thread loopPathProcessing {loopPath};             // 高频位姿
    

    ros::spin();

    return 0;
}