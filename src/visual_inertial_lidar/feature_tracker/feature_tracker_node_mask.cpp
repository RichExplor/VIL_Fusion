#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Bool.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>

#include "feature_tracker.h"

#include<mutex>
#include<thread>

#define SHOW_UNDISTORTION 0

vector<uchar> r_status;
vector<float> r_err;

queue<sensor_msgs::ImageConstPtr> img_buf;
queue<sensor_msgs::ImageConstPtr> mask_buf;

ros::Publisher pub_img,pub_match;
ros::Publisher pub_restart;

FeatureTracker trackerData[NUM_OF_CAM];
double first_image_time;
int pub_count = 1;
bool first_image_flag = true;
double last_image_time = 0;
bool init_pub = 0;

mutex m_buf;

double timeMask = 0;
double timeImage = 0;


void image_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    m_buf.lock();
    img_buf.push(img_msg);
    m_buf.unlock();
}

void mask_callback(const sensor_msgs::ImageConstPtr &mask_msg)
{
    m_buf.lock();
    mask_buf.push(mask_msg);
    m_buf.unlock();
}


void img_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    if(first_image_flag)
    {
        first_image_flag = false;
        first_image_time = img_msg->header.stamp.toSec();
        last_image_time = img_msg->header.stamp.toSec();
        return;
    }
    // detect unstable camera stream
    if (img_msg->header.stamp.toSec() - last_image_time > 1.0 || img_msg->header.stamp.toSec() < last_image_time)
    {
        ROS_WARN("image discontinue! reset the feature tracker!");
        first_image_flag = true; 
        last_image_time = 0;
        pub_count = 1;
        std_msgs::Bool restart_flag;
        restart_flag.data = true;
        pub_restart.publish(restart_flag);
        return;
    }
    last_image_time = img_msg->header.stamp.toSec();
    // frequency control
    if (round(1.0 * pub_count / (img_msg->header.stamp.toSec() - first_image_time)) <= FREQ)
    {
        PUB_THIS_FRAME = true;
        // reset the frequency control
        if (abs(1.0 * pub_count / (img_msg->header.stamp.toSec() - first_image_time) - FREQ) < 0.01 * FREQ)
        {
            first_image_time = img_msg->header.stamp.toSec();
            pub_count = 0;
        }
    }
    else
        PUB_THIS_FRAME = false;

    cv_bridge::CvImageConstPtr ptr;
    if (img_msg->encoding == "8UC1")
    {
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    }
    else
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

    cv::Mat show_img = ptr->image;
    TicToc t_r;
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ROS_DEBUG("processing camera %d", i);
        cout<<"processing camera: "<<i<<endl;
        if (i != 1 || !STEREO_TRACK)
            trackerData[i].readImage(ptr->image.rowRange(ROW * i, ROW * (i + 1)), img_msg->header.stamp.toSec());
        else
        {
            if (EQUALIZE)
            {
                cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
                clahe->apply(ptr->image.rowRange(ROW * i, ROW * (i + 1)), trackerData[i].cur_img);
            }
            else
                trackerData[i].cur_img = ptr->image.rowRange(ROW * i, ROW * (i + 1));
        }

#if SHOW_UNDISTORTION
        trackerData[i].showUndistortion("undistrotion_" + std::to_string(i));
#endif
    }

    for (unsigned int i = 0;; i++)
    {
        bool completed = false;
        for (int j = 0; j < NUM_OF_CAM; j++)
            if (j != 1 || !STEREO_TRACK)
                completed |= trackerData[j].updateID(i);
        if (!completed)
            break;
    }

   if (PUB_THIS_FRAME)
   {
        sensor_msgs::PointCloudPtr feature_points(new sensor_msgs::PointCloud);
        sensor_msgs::ChannelFloat32 id_of_point;
        sensor_msgs::ChannelFloat32 u_of_point;
        sensor_msgs::ChannelFloat32 v_of_point;
        sensor_msgs::ChannelFloat32 velocity_x_of_point;
        sensor_msgs::ChannelFloat32 velocity_y_of_point;

        feature_points->header = img_msg->header;
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
                    velocity_x_of_point.values.push_back(0);
                    velocity_y_of_point.values.push_back(0);
                }
            }
        }
        feature_points->channels.push_back(id_of_point);
        feature_points->channels.push_back(u_of_point);
        feature_points->channels.push_back(v_of_point);
        feature_points->channels.push_back(velocity_x_of_point);
        feature_points->channels.push_back(velocity_y_of_point);
        ROS_DEBUG("publish %f, at %f", feature_points->header.stamp.toSec(), ros::Time::now().toSec());
        // skip the first image; since no optical speed on frist image
        if (!init_pub)
        {
            init_pub = 1;
        }
        else
        {
            // std::cout<<feature_points->header.stamp.toSec()<<std::endl;
            ROS_INFO("publish: %f", feature_points->header.stamp.toSec());
            pub_img.publish(feature_points);
        }     
    }
    ROS_INFO("whole feature tracker processing costs: %f", t_r.toc());
}


void imageMaskProcess(const sensor_msgs::ImageConstPtr &img_msg, const sensor_msgs::ImageConstPtr &mask_msg)
{
    if(first_image_flag)
    {
        first_image_flag = false;
        first_image_time = img_msg->header.stamp.toSec();
        last_image_time = img_msg->header.stamp.toSec();
        return;
    }
    // detect unstable camera stream
    if (img_msg->header.stamp.toSec() - last_image_time > 1.0 || img_msg->header.stamp.toSec() < last_image_time)
    {
        ROS_WARN("image discontinue! reset the feature tracker!");
        first_image_flag = true; 
        last_image_time = 0;
        pub_count = 1;
        std_msgs::Bool restart_flag;
        restart_flag.data = true;
        pub_restart.publish(restart_flag);
        return;
    }
    last_image_time = img_msg->header.stamp.toSec();
    // frequency control
    if (round(1.0 * pub_count / (img_msg->header.stamp.toSec() - first_image_time)) <= FREQ)
    {
        PUB_THIS_FRAME = true;
        // reset the frequency control
        if (abs(1.0 * pub_count / (img_msg->header.stamp.toSec() - first_image_time) - FREQ) < 0.01 * FREQ)
        {
            first_image_time = img_msg->header.stamp.toSec();
            pub_count = 0;
        }
    }
    else
        PUB_THIS_FRAME = false;

    cv_bridge::CvImageConstPtr ptr;
    if (img_msg->encoding == "8UC1")
    {
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    }
    else
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

    cv::Mat show_img = ptr->image;


    cv_bridge::CvImageConstPtr ptr_mask;
    if (mask_msg->encoding == "8UC1")
    {
        sensor_msgs::Image img;
        img.header = mask_msg->header;
        img.height = mask_msg->height;
        img.width = mask_msg->width;
        img.is_bigendian = mask_msg->is_bigendian;
        img.step = mask_msg->step;
        img.data = mask_msg->data;
        img.encoding = "mono8";
        ptr_mask = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    }
    else
        ptr_mask = cv_bridge::toCvCopy(mask_msg, sensor_msgs::image_encodings::MONO8);

    cv::Mat mask_temp = ptr_mask->image;

    cv::Mat show_mask, mask_white;

    mask_white = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));  // 定义一个白色背景

    mask_white.copyTo(show_mask, mask_temp);  // 使用减法操作，将图像的mask显示出来，其中show_mask是得到的结果（输出）


    TicToc t_r;
    // Dilation settings
    int dilation_size = 5;
    cv::Mat kernel = getStructuringElement(cv::MORPH_ELLIPSE,
                                           cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                           cv::Point( dilation_size, dilation_size ) );

    cv::Mat maskRCNN;
    cv::Mat maskRCNNdil = show_mask.clone();
    cv::erode(show_mask, maskRCNNdil, kernel);  // 由于背景时白色，因此采用腐蚀操作，使mask更大。否则采用膨胀操作

    // cv::imshow("img_show", show_img);
    // cv::namedWindow("mask_show", 1);
    // cv::imshow("mask_show", maskRCNNdil);
    // cv::waitKey(1);

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ROS_DEBUG("processing camera %d", i);
        if (i != 1 || !STEREO_TRACK)
            trackerData[i].readImage_mask(show_img.rowRange(ROW * i, ROW * (i + 1)), maskRCNNdil.rowRange(ROW * i, ROW * (i + 1)), img_msg->header.stamp.toSec()); // 单目图像 + mask图像(白色背景)
            // trackerData[i].readImage(ptr->image.rowRange(ROW * i, ROW * (i + 1)), img_msg->header.stamp.toSec());
        else
        {
            if (EQUALIZE)
            {
                cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
                clahe->apply(ptr->image.rowRange(ROW * i, ROW * (i + 1)), trackerData[i].cur_img);
            }
            else
                trackerData[i].cur_img = ptr->image.rowRange(ROW * i, ROW * (i + 1));
        }

#if SHOW_UNDISTORTION
        trackerData[i].showUndistortion("undistrotion_" + std::to_string(i));
#endif
    }

    // trackerData[0].readImage_mask(show_img, maskRCNNdil, img_msg->header.stamp.toSec()); // 单目图像 + mask图像(白色背景)
    // trackerData[0].readImage(show_img, img_msg->header.stamp.toSec()); // 单目图像 + mask图像(白色背景)


    for (unsigned int i = 0;; i++)
    {
        bool completed = false;
        for (int j = 0; j < NUM_OF_CAM; j++)
            if (j != 1 || !STEREO_TRACK)
                completed |= trackerData[j].updateID(i);
        if (!completed)
            break;
    }

   if (PUB_THIS_FRAME)
   {
        sensor_msgs::PointCloudPtr feature_points(new sensor_msgs::PointCloud);
        sensor_msgs::ChannelFloat32 id_of_point;
        sensor_msgs::ChannelFloat32 u_of_point;
        sensor_msgs::ChannelFloat32 v_of_point;
        sensor_msgs::ChannelFloat32 velocity_x_of_point;
        sensor_msgs::ChannelFloat32 velocity_y_of_point;

        feature_points->header = img_msg->header;
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
                    velocity_x_of_point.values.push_back(0);
                    velocity_y_of_point.values.push_back(0);
                }
            }
        }
        feature_points->channels.push_back(id_of_point);
        feature_points->channels.push_back(u_of_point);
        feature_points->channels.push_back(v_of_point);
        feature_points->channels.push_back(velocity_x_of_point);
        feature_points->channels.push_back(velocity_y_of_point);

        sensor_msgs::ChannelFloat32 depth_points;
        depth_points.name = "depth";
        depth_points.values.resize(feature_points->points.size(), -1);
        feature_points->channels.push_back(depth_points);

        ROS_DEBUG("publish %f, at %f", feature_points->header.stamp.toSec(), ros::Time::now().toSec());
        // skip the first image; since no optical speed on frist image
        if (!init_pub)
        {
            init_pub = 1;
        }
        else
        {
            // std::cout<<feature_points->header.stamp.toSec()<<std::endl;
            ROS_INFO("publish: %f", feature_points->header.stamp.toSec());
            pub_img.publish(feature_points);
        }     
    }
    if (SHOW_TRACK)
    {
        ptr = cv_bridge::cvtColor(ptr, sensor_msgs::image_encodings::BGR8);
        cv::Mat stereo_img = ptr->image;

        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            cv::Mat tmp_img = stereo_img.rowRange(i * ROW, (i + 1) * ROW);
            cv::cvtColor(show_img, tmp_img, CV_GRAY2RGB);

            for (unsigned int j = 0; j < trackerData[i].cur_pts.size(); j++)
            {
                double len = std::min(1.0, 1.0 * trackerData[i].track_cnt[j] / WINDOW_SIZE);
                cv::circle(tmp_img, trackerData[i].cur_pts[j], 2, cv::Scalar(0, 0, 255), 2);  // 越红说明跟踪的越长
            }
        }
        // cv::imshow("vis", stereo_img);
        // cv::waitKey(5);
        pub_match.publish(ptr->toImageMsg());
    }
    ROS_INFO("whole feature tracker processing costs: %f ms", t_r.toc());
}


// thread: 联合处理动态特征点
void process()
{
    while (1)
    {

        std::unique_lock<std::mutex> lk(m_buf);
        while(!img_buf.empty() && !mask_buf.empty())
        {
            while(!img_buf.empty() && img_buf.front()->header.stamp.toSec() < mask_buf.front()->header.stamp.toSec())
                img_buf.pop();
            if(img_buf.empty())
            {
                break;
            }

            while(!mask_buf.empty() && mask_buf.front()->header.stamp.toSec() < img_buf.front()->header.stamp.toSec())
                mask_buf.pop();
            if(mask_buf.empty())
            {
                break;
            }

            timeMask = mask_buf.front()->header.stamp.toSec();
            timeImage = img_buf.front()->header.stamp.toSec();

            if(timeMask != timeImage)
            {
                printf("time unsync message!");
                break;
            }

            sensor_msgs::ImageConstPtr mask_image = mask_buf.front();
            mask_buf.pop();
            sensor_msgs::ImageConstPtr img_image = img_buf.front();
            img_buf.pop();

            imageMaskProcess(img_image, mask_image);  // 联合处理

        }

    }
}



int main(int argc, char **argv)
{
    ros::init(argc, argv, "sensor_fusion_feature");
    ros::NodeHandle n("~");
    // ros::NodeHandle n;
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


    printf("load is successful!!!!! \n");

    ros::Subscriber sub_img = n.subscribe<sensor_msgs::Image>("/feature_tracker_mask/image_img", 1000, image_callback);
    ros::Subscriber mask_img = n.subscribe<sensor_msgs::Image>("/feature_tracker_mask/image_mask", 1000, mask_callback);

    pub_img = n.advertise<sensor_msgs::PointCloud>("feature", 1000);
    pub_match = n.advertise<sensor_msgs::Image>("feature_img",1000);
    pub_restart = n.advertise<std_msgs::Bool>("restart",1000);

    std::thread feature_process{process};

    ros::spin();

    return 0;
}

