#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Bool.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>

#include "feature_tracker.h"

#include "include/featureExtraction.hpp"
#include "include/EstimationMapping.hpp"

#define SHOW_UNDISTORTION 0

featureExtraction featureExtractFactor;
EstimationMapping Estimator;

Eigen::Quaterniond q_last;
Eigen::Vector3d t_last;

vector<uchar> r_status;
vector<float> r_err;
queue<sensor_msgs::ImageConstPtr> img_buf;
queue<sensor_msgs::PointCloud2ConstPtr> lidar_buf;

mutex mBuf;

ros::Publisher pub_img,pub_match;
ros::Publisher pub_restart;

ros::Publisher pub_Odometry, pub_GlobalPath, pub_GlobalMap, pub_DepthImg;
nav_msgs::Path laserPath;

FeatureTracker trackerData[NUM_OF_CAM];
double first_image_time;
int pub_count = 1;
bool first_image_flag = true;
double last_image_time = 0;
bool init_pub = 0;


sensor_msgs::PointCloud2ConstPtr laserCloudMsgs;
sensor_msgs::ImageConstPtr imageMsgs;

double timeLaserCloud = 0;
double timeImage = 0;

const int num_bins = 360;
std::vector<std::vector<PointType>> pointsArray;

sensor_msgs::ChannelFloat32 getFeatureDepth(pcl::PointCloud<PointType>::Ptr& depth_cloud_local, cv::Mat show_img, std::vector<geometry_msgs::Point32> features_2d)
{
    sensor_msgs::ChannelFloat32 depth_points;
    depth_points.name = "depth";
    depth_points.values.resize(features_2d.size(), -1);

    // 4.1 投影特征点到球体归一化坐标系
    pcl::PointCloud<PointType>::Ptr features_3d_sphere(new pcl::PointCloud<PointType>());
    for (size_t i = 0; i < features_2d.size(); ++i)
    {
        // 归一化坐标系到球体坐标系
        Eigen::Vector3f feature_cur(features_2d[i].x, features_2d[i].y, features_2d[i].z); // z always equal to 1
        feature_cur.normalize(); 
        // 转换为 ROS 坐标系
        PointType p;
        p.x = feature_cur(0);
        p.y = feature_cur(1);
        p.z = feature_cur(2);
        p.intensity = -1; // 用于保存深度
        features_3d_sphere->push_back(p);
    }


    // 4.2 投影深度点云到归一化球体坐标系
    float bin_res = 180.0 / (float)num_bins; 
    pcl::PointCloud<PointType>::Ptr depth_cloud_unit_sphere(new pcl::PointCloud<PointType>());
    for (size_t i = 0; i < depth_cloud_local->size(); ++i)
    {
        PointType p = depth_cloud_local->points[i];
        float range = pointDistance(p);
        p.x /= range;
        p.y /= range;
        p.z /= range;
        p.intensity = range;
        depth_cloud_unit_sphere->push_back(p);
    }

    if (depth_cloud_unit_sphere->size() < 10)
    {
        std::cout<<"depth cloud is too faw!"<<std::endl;
        return depth_points;
    }
    
    // 4.3 利用稀疏点云创建kd-tree，并查找特征点最近邻点云的深度
    pcl::KdTreeFLANN<PointType>::Ptr kdtree(new pcl::KdTreeFLANN<PointType>());
    kdtree->setInputCloud(depth_cloud_unit_sphere);

    std::vector<int> pointSearchInd;
    std::vector<float> pointSearchSqDis;
    float dist_sq_threshold = pow(sin(bin_res / 180.0 * M_PI) * 5.0, 2);
    for (size_t i = 0; i < features_3d_sphere->size(); ++i)
    {
        kdtree->nearestKSearch(features_3d_sphere->points[i], 3, pointSearchInd, pointSearchSqDis);
        if (pointSearchInd.size() == 3 && pointSearchSqDis[2] < dist_sq_threshold)
        {
            float r1 = depth_cloud_unit_sphere->points[pointSearchInd[0]].intensity;
            Eigen::Vector3f A(depth_cloud_unit_sphere->points[pointSearchInd[0]].x * r1,
                            depth_cloud_unit_sphere->points[pointSearchInd[0]].y * r1,
                            depth_cloud_unit_sphere->points[pointSearchInd[0]].z * r1);

            float r2 = depth_cloud_unit_sphere->points[pointSearchInd[1]].intensity;
            Eigen::Vector3f B(depth_cloud_unit_sphere->points[pointSearchInd[1]].x * r2,
                            depth_cloud_unit_sphere->points[pointSearchInd[1]].y * r2,
                            depth_cloud_unit_sphere->points[pointSearchInd[1]].z * r2);

            float r3 = depth_cloud_unit_sphere->points[pointSearchInd[2]].intensity;
            Eigen::Vector3f C(depth_cloud_unit_sphere->points[pointSearchInd[2]].x * r3,
                            depth_cloud_unit_sphere->points[pointSearchInd[2]].y * r3,
                            depth_cloud_unit_sphere->points[pointSearchInd[2]].z * r3);

            // https://math.stackexchange.com/questions/100439/determine-where-a-vector-will-intersect-a-plane
            Eigen::Vector3f V(features_3d_sphere->points[i].x,
                            features_3d_sphere->points[i].y,
                            features_3d_sphere->points[i].z);

            Eigen::Vector3f N = (A - B).cross(B - C);
            float s = (N(0) * A(0) + N(1) * A(1) + N(2) * A(2)) 
                    / (N(0) * V(0) + N(1) * V(1) + N(2) * V(2));

            float min_depth = std::min(r1, std::min(r2, r3));
            float max_depth = std::max(r1, std::max(r2, r3));
            if (max_depth - min_depth > 2 || s <= 0.5)
            {
                continue;
            } else if (s - max_depth > 0) {
                s = max_depth;
            } else if (s - min_depth < 0) {
                s = min_depth;
            }
            // 如果深度可靠，保存
            features_3d_sphere->points[i].x *= s;
            features_3d_sphere->points[i].y *= s;
            features_3d_sphere->points[i].z *= s;
            // the obtained depth here is for unit sphere, VINS estimator needs depth for normalized feature (by value z), (lidar x = camera z)
            features_3d_sphere->points[i].intensity = features_3d_sphere->points[i].z;
        }
    }

    // 4.4 更新深度并返回
    int countDepth = 0;
    for (size_t i = 0; i < features_3d_sphere->size(); ++i)
    {
        if (features_3d_sphere->points[i].intensity > 2.0)  // 3.0
        {
            depth_points.values[i] = features_3d_sphere->points[i].intensity;
            countDepth++;
        }
    }

    ROS_DEBUG("depth feature count is %d ", countDepth);
    
    // 4.5 将3d点变换为2d点
    vector<cv::Point2d> points_2f;
    vector<float> points_distance;

    for (size_t i = 0; i < depth_cloud_local->points.size(); ++i)
    {
        // 转换图像坐标系
        Eigen::Vector3d p_3d(depth_cloud_local->points[i].x, depth_cloud_local->points[i].y, depth_cloud_local->points[i].z);
        Eigen::Vector2d p_2d;
        trackerData[0].m_camera->spaceToPlane(p_3d, p_2d);

        points_2f.push_back(cv::Point2f(p_2d(0), p_2d(1)));
        points_distance.push_back( pointDistance(depth_cloud_local->points[i]) );
    }

    cv::Mat showImage, circleImage;
    cv::cvtColor(show_img, showImage, cv::COLOR_GRAY2RGB);
    circleImage = showImage.clone();
    for (size_t i = 0; i < points_2f.size(); ++i)
    {
        float r, g, b;
        getColor(points_distance[i], 20, r, g, b);
        cv::circle(circleImage, points_2f[i], 0, cv::Scalar(r, g, b), 2);
    }
    cv::addWeighted(showImage, 1.0, circleImage, 0.7, 0, showImage);

    cv_bridge::CvImage bridge;
    bridge.image = showImage;
    bridge.encoding = "rgb8";
    sensor_msgs::Image::Ptr imageShowPtr = bridge.toImageMsg();
    imageShowPtr->header.stamp = imageMsgs->header.stamp;
    pub_DepthImg.publish(imageShowPtr); // 发布点云

    return depth_points;
}


void img_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    mBuf.lock();
    img_buf.push(img_msg);
    mBuf.unlock();
}

void lidar_callback(const sensor_msgs::PointCloud2ConstPtr &lidarMsgs)
{
    mBuf.lock();
    lidar_buf.push(lidarMsgs);
    mBuf.unlock();
}

void processing()
{
    while(1)
    {
        while(!img_buf.empty() && !lidar_buf.empty())
        {
            mBuf.lock();
            // 0.03 激光和相机的时间差最大 0.03
            std::cout.precision(19);
            while( !img_buf.empty() && ( lidar_buf.front()->header.stamp.toSec() - img_buf.front()->header.stamp.toSec() ) > 0.03)
            {
                // std::cout<<"image time is too slow : "<<img_buf.front()->header.stamp.toSec()<<std::endl;
                img_buf.pop();
            }
            if(img_buf.empty())
            {
                mBuf.unlock();
                break;
            }

            while( !lidar_buf.empty() && ( img_buf.front()->header.stamp.toSec() - lidar_buf.front()->header.stamp.toSec() ) > 0.03)
            {
                // std::cout<<"lidar time is too slow : "<<lidar_buf.front()->header.stamp.toSec()<<std::endl;
                lidar_buf.pop();
            }
            if(lidar_buf.empty())
            {
                mBuf.unlock();
                break;
            }

            timeLaserCloud = lidar_buf.front()->header.stamp.toSec();
            timeImage = img_buf.front()->header.stamp.toSec();
            
            if(abs(timeLaserCloud - timeImage) > 0.03)
            {
                std::cout<<"unsync message!"<<std::endl;
                mBuf.unlock();
                break;
            }

            laserCloudMsgs = lidar_buf.front();
            lidar_buf.pop();

            imageMsgs = img_buf.front();
            img_buf.pop();

            mBuf.unlock();

            cv_bridge::CvImageConstPtr ptr;
            if (imageMsgs->encoding == "8UC1")
            {
                sensor_msgs::Image img;
                img.header = imageMsgs->header;
                img.height = imageMsgs->height;
                img.width = imageMsgs->width;
                img.is_bigendian = imageMsgs->is_bigendian;
                img.step = imageMsgs->step;
                img.data = imageMsgs->data;
                img.encoding = "mono8";
                ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
            }
            else
                ptr = cv_bridge::toCvCopy(imageMsgs, sensor_msgs::image_encodings::MONO8);
            
            cv::Mat show_img = ptr->image;

            PUB_THIS_FRAME = true;
            // 0.图像特征提取
            trackerData[0].readImage(ptr->image.rowRange(0, ROW), imageMsgs->header.stamp.toSec());
            for (unsigned int i = 0;; i++)
            {
                bool completed = false;
                completed |= trackerData[0].updateID(i);
                if (!completed)
                    break;
            }

            sensor_msgs::PointCloudPtr feature_points(new sensor_msgs::PointCloud);
            sensor_msgs::ChannelFloat32 id_of_point;
            sensor_msgs::ChannelFloat32 u_of_point;
            sensor_msgs::ChannelFloat32 v_of_point;
            sensor_msgs::ChannelFloat32 velocity_x_of_point;
            sensor_msgs::ChannelFloat32 velocity_y_of_point;

            feature_points->header.stamp = imageMsgs->header.stamp;
            feature_points->header.frame_id = "world";

            vector<set<int>> hash_ids(NUM_OF_CAM);
            for (int i = 0; i < NUM_OF_CAM; i++)
            {
                auto &un_pts = trackerData[i].cur_un_pts;
                auto &cur_pts = trackerData[i].cur_pts;
                auto &ids = trackerData[i].ids;
                auto &pts_velocity = trackerData[i].pts_velocity;
                for (unsigned int j = 0; j < ids.size(); j++)
                {
                    if (trackerData[i].track_cnt[j] > 1)
                    {
                        int p_id = ids[j];
                        hash_ids[i].insert(p_id);
                        geometry_msgs::Point32 p;
                        p.x = un_pts[j].x;
                        p.y = un_pts[j].y;
                        p.z = 1;

                        feature_points->points.push_back(p);
                        id_of_point.values.push_back(p_id * NUM_OF_CAM + i);
                        u_of_point.values.push_back(cur_pts[j].x);
                        v_of_point.values.push_back(cur_pts[j].y);
                        velocity_x_of_point.values.push_back(pts_velocity[j].x);
                        velocity_y_of_point.values.push_back(pts_velocity[j].y);
                    }
                }
            }

            feature_points->channels.push_back(id_of_point);
            feature_points->channels.push_back(u_of_point);
            feature_points->channels.push_back(v_of_point);
            feature_points->channels.push_back(velocity_x_of_point);
            feature_points->channels.push_back(velocity_y_of_point);

            // 1.转换点云消息为pcl
            pcl::PointCloud<PointType>::Ptr PointCloudFull(new pcl::PointCloud<PointType>());
            pcl::fromROSMsg(*laserCloudMsgs, *PointCloudFull);  // 转换消息类型

           // 1.1 点云特征提取--边缘点/平面点
           pcl::PointCloud<PointType>::Ptr MapCloud(new pcl::PointCloud<PointType>());
           pcl::PointCloud<PointType>::Ptr featureCloud_Edge(new pcl::PointCloud<PointType>());
           pcl::PointCloud<PointType>::Ptr featureCloud_Surf(new pcl::PointCloud<PointType>());
           featureExtractFactor.extractFeature(PointCloudFull, featureCloud_Edge, featureCloud_Surf);

            // 2.滤除掉相机视野之外的点云
            pcl::PointCloud<PointType>::Ptr PointCloudFilter(new pcl::PointCloud<PointType>());
            for(size_t i = 0; i < PointCloudFull->size(); ++i)
            {
                PointType p = PointCloudFull->points[i];
                if (p.x > 0 && abs(p.y/p.x) <= 10 && abs(p.z/p.x) <= 10)  // arctan(10) = 84°
                    PointCloudFilter->push_back(p);
            }
            *PointCloudFull = *PointCloudFilter;

            // 3.转换到相机坐标系下
            pcl::PointCloud<PointType>::Ptr PointCloudOffset(new pcl::PointCloud<PointType>());
            pcl::transformPointCloud(*PointCloudFull, *PointCloudOffset, LIDAR_CAMERA_EX);

            *PointCloudFull = *PointCloudOffset;

            // 4.获取特征点深度
            sensor_msgs::ChannelFloat32 depth_points = getFeatureDepth(PointCloudFull, show_img, feature_points->points);
            feature_points->channels.push_back(depth_points);

            // 5. 跳过第一帧，发布特征点信息和画出带有深度的特征
            if (!init_pub)
            {
                init_pub = 1;

                Estimator.localMapInited(featureCloud_Edge, featureCloud_Surf);
                cout<<"system is inited."<<endl;

                q_last = Eigen::Quaterniond(1,0,0,0);
                t_last = Eigen::Vector3d(0,0,0);

                continue;
            }
            else
            {
                // 点/面特征约束优化
                Estimator.optimation_processing(featureCloud_Edge, featureCloud_Surf);
                Estimator.getMapCloud(MapCloud);

                // publish current pose
                Eigen::Quaterniond q_estimator(Estimator.globalOdom.rotation());
                Eigen::Vector3d t_estimator = Estimator.globalOdom.translation();

                // publish tf
                static tf::TransformBroadcaster br;
                tf::Transform transform;
                transform.setOrigin( tf::Vector3(t_estimator.x(), t_estimator.y(), t_estimator.z()) );
                tf::Quaternion q_tf(q_estimator.x(), q_estimator.y(), q_estimator.z(), q_estimator.w());
                transform.setRotation(q_tf);
                br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "world", "body"));

                // 计算相对位姿，用于发布, T_ij = [R_ij, t_ij], 将上一帧变换到当前帧的相对变换矩阵
                Eigen::Quaterniond q_relative = q_last.inverse() * q_estimator;
                Eigen::Vector3d t_relative = q_last.inverse() * (t_estimator - t_last);

                // publish Odometry and Path
                nav_msgs::Odometry laserOdometry;
                laserOdometry.header.frame_id = "world";
                laserOdometry.child_frame_id  = "body";
                laserOdometry.header.stamp = imageMsgs->header.stamp;
                laserOdometry.pose.pose.orientation.x = q_relative.x();
                laserOdometry.pose.pose.orientation.y = q_relative.y();
                laserOdometry.pose.pose.orientation.z = q_relative.z();
                laserOdometry.pose.pose.orientation.w = q_relative.w();
                laserOdometry.pose.pose.position.x    = t_relative.x();
                laserOdometry.pose.pose.position.y    = t_relative.y();
                laserOdometry.pose.pose.position.z    = t_relative.z();
                pub_Odometry.publish(laserOdometry);

                // publish Odometry and Path
                nav_msgs::Odometry laserOdometryPath;
                laserOdometryPath.header.frame_id = "world";
                laserOdometryPath.child_frame_id  = "body";
                laserOdometryPath.header.stamp = imageMsgs->header.stamp;
                laserOdometryPath.pose.pose.orientation.x = q_estimator.x();
                laserOdometryPath.pose.pose.orientation.y = q_estimator.y();
                laserOdometryPath.pose.pose.orientation.z = q_estimator.z();
                laserOdometryPath.pose.pose.orientation.w = q_estimator.w();
                laserOdometryPath.pose.pose.position.x    = t_estimator.x();
                laserOdometryPath.pose.pose.position.y    = t_estimator.y();
                laserOdometryPath.pose.pose.position.z    = t_estimator.z();

                geometry_msgs::PoseStamped laserPose;
                laserPose.header = laserOdometryPath.header;
                laserPose.pose   = laserOdometryPath.pose.pose;
                laserPath.header.stamp = laserOdometryPath.header.stamp;
                laserPath.header.frame_id = "world";
                laserPath.poses.push_back(laserPose);
                pub_GlobalPath.publish(laserPath);

                // publish local map
                sensor_msgs::PointCloud2 MapCloudMsg;
                pcl::toROSMsg(*MapCloud, MapCloudMsg);
                MapCloudMsg.header.stamp = imageMsgs->header.stamp;;
                MapCloudMsg.header.frame_id = "world";
                pub_GlobalMap.publish(MapCloudMsg);

                q_last = q_estimator;
                t_last = t_estimator;

                // 发布视觉特征点
                pub_img.publish(feature_points);
            }
            if (SHOW_TRACK)
            {
                ptr = cv_bridge::cvtColor(ptr, sensor_msgs::image_encodings::BGR8);
                cv::Mat stereo_img = ptr->image;

                for (int i = 0; i < NUM_OF_CAM; i++)
                {
                    cv::Mat tmp_img = stereo_img.rowRange(i * ROW, (i + 1) * ROW);
                    cv::cvtColor(show_img, tmp_img, CV_GRAY2RGB);

                    // 按照深度画出点
                    for (unsigned int j = 0; j < feature_points->channels[5].values.size(); j++)
                    {
                        //draw feature point
                        if(feature_points->channels[5].values[j] == -1)
                            cv::circle(tmp_img, cv::Point(feature_points->channels[1].values[j], feature_points->channels[2].values[j]), 2, cv::Scalar(255, 0, 255), 2);
                        else
                            cv::circle(tmp_img, cv::Point(feature_points->channels[1].values[j], feature_points->channels[2].values[j]), 2, cv::Scalar(0, 255, 255), 2);
                    }
                }
                pub_match.publish(ptr->toImageMsg());
            }

        }
        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "sensor_fusion_feature");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    readParameters(n);

    for (int i = 0; i < NUM_OF_CAM; i++)
        trackerData[i].readIntrinsicParameter(CAM_NAMES[i]);

    if(FISHEYE)
    {
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            trackerData[i].fisheye_mask = cv::imread(FISHEYE_MASK, 0);
            if(!trackerData[i].fisheye_mask.data)
            {
                ROS_INFO("load mask fail");
                ROS_BREAK();
            }
            else
                ROS_INFO("load mask success");
        }
    }

    pointsArray.resize(num_bins);
    for (int i = 0; i < num_bins; ++i)
        pointsArray[i].resize(num_bins);

    featureExtractFactor.initParam(n);
    Estimator.initParameter(n);

    ros::Subscriber sub_img   = n.subscribe(IMAGE_TOPIC, 100, img_callback);
    ros::Subscriber sub_lidar = n.subscribe(LIDAR_TOPIC, 100, lidar_callback);

    pub_img = n.advertise<sensor_msgs::PointCloud>("feature", 1000);
    pub_match = n.advertise<sensor_msgs::Image>("feature_img",1000);
    pub_restart = n.advertise<std_msgs::Bool>("restart",1000);

    pub_DepthImg = n.advertise<sensor_msgs::Image>("depth_img", 1000);
    pub_Odometry = n.advertise<nav_msgs::Odometry>("/Odometry", 1000);
    pub_GlobalPath = n.advertise<nav_msgs::Path>("/path", 1000);
    pub_GlobalMap  = n.advertise<sensor_msgs::PointCloud2>("/GlobalMap", 1000);

    std::thread feature_thread{processing};

    ros::spin();
    return 0;
}