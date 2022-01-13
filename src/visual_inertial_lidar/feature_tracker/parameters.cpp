#include "parameters.h"

std::string IMAGE_TOPIC;
std::string IMU_TOPIC;
std::string LIDAR_TOPIC;
std::vector<std::string> CAM_NAMES;
std::string FISHEYE_MASK;
int MAX_CNT;
int MIN_DIST;
int WINDOW_SIZE;
int FREQ;
double F_THRESHOLD;
int SHOW_TRACK;
int STEREO_TRACK;
int EQUALIZE;
int ROW;
int COL;
int FOCAL_LENGTH;
int FISHEYE;
bool PUB_THIS_FRAME;

Eigen::Matrix4d LIDAR_CAMERA_EX;

template <typename T>
T readParam(ros::NodeHandle &n, std::string name)
{
    T ans;
    if (n.getParam(name, ans))
    {
        ROS_INFO_STREAM("Loaded " << name << ": " << ans);
    }
    else
    {
        ROS_ERROR_STREAM("Failed to load " << name);
        n.shutdown();
    }
    return ans;
}

void readParameters(ros::NodeHandle &n)
{
    std::string config_file;
    config_file = readParam<std::string>(n, "config_file");
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }
    std::string VINS_FOLDER_PATH = readParam<std::string>(n, "vins_folder");

    fsSettings["image_topic"] >> IMAGE_TOPIC;
    fsSettings["imu_topic"] >> IMU_TOPIC;
    fsSettings["lidar_topic"] >> LIDAR_TOPIC;

    cv::Mat cv_R, cv_T;
    fsSettings["exRotation"] >> cv_R;
    fsSettings["exTranslation"] >> cv_T;
    Eigen::Matrix3d eigen_R;
    Eigen::Vector3d eigen_T;
    cv::cv2eigen(cv_R, eigen_R);
    cv::cv2eigen(cv_T, eigen_T);

    Eigen::Quaterniond Q(eigen_R);
    eigen_R = Q.normalized();
    LIDAR_CAMERA_EX = Eigen::Matrix4d::Identity();
    LIDAR_CAMERA_EX.block(0,0,3,3) = eigen_R;
    LIDAR_CAMERA_EX.block(0,3,3,1) = eigen_T;

    std::cout<<"LIDAR_CAMERA_EX = "<<LIDAR_CAMERA_EX<<std::endl;

    MAX_CNT = fsSettings["max_cnt"];
    MIN_DIST = fsSettings["min_dist"];
    ROW = fsSettings["image_height"];
    COL = fsSettings["image_width"];
    FREQ = fsSettings["freq"];
    F_THRESHOLD = fsSettings["F_threshold"];
    SHOW_TRACK = fsSettings["show_track"];
    EQUALIZE = fsSettings["equalize"];
    FISHEYE = fsSettings["fisheye"];
    if (FISHEYE == 1)
        FISHEYE_MASK = VINS_FOLDER_PATH + "config/fisheye_mask.jpg";
    CAM_NAMES.push_back(config_file);

    WINDOW_SIZE = 20;
    STEREO_TRACK = false;
    FOCAL_LENGTH = 460;
    PUB_THIS_FRAME = false;

    if (FREQ == 0)
        FREQ = 100;

    fsSettings.release();


}

float pointDistance(PointType p)
{
    return sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
}

void publishCloud(ros::Publisher *thisPub, pcl::PointCloud<PointType>::Ptr thisCloud, ros::Time thisStamp, std::string thisFrame)
{
    if (thisPub->getNumSubscribers() == 0)
        return;
    sensor_msgs::PointCloud2 tempCloud;
    pcl::toROSMsg(*thisCloud, tempCloud);
    tempCloud.header.stamp = thisStamp;
    tempCloud.header.frame_id = thisFrame;
    thisPub->publish(tempCloud); 
}

void getColor(float p, float np, float&r, float&g, float&b) 
{
    float inc = 6.0 / np;
    float x = p * inc;
    r = 0.0f; g = 0.0f; b = 0.0f;
    if ((0 <= x && x <= 1) || (5 <= x && x <= 6)) r = 1.0f;
    else if (4 <= x && x <= 5) r = x - 4;
    else if (1 <= x && x <= 2) r = 1.0f - (x - 1);

    if (1 <= x && x <= 3) g = 1.0f;
    else if (0 <= x && x <= 1) g = x - 0;
    else if (3 <= x && x <= 4) g = 1.0f - (x - 3);

    if (3 <= x && x <= 5) b = 1.0f;
    else if (2 <= x && x <= 3) b = x - 2;
    else if (5 <= x && x <= 6) b = 1.0f - (x - 5);
    r *= 255.0;
    g *= 255.0;
    b *= 255.0;
}

