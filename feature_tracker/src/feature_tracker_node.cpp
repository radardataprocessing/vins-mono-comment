#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Bool.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>

#include "feature_tracker.h"

#define SHOW_UNDISTORTION 0

vector<uchar> r_status;
vector<float> r_err;
queue<sensor_msgs::ImageConstPtr> img_buf;

ros::Publisher pub_img,pub_match;
ros::Publisher pub_restart;

FeatureTracker trackerData[NUM_OF_CAM];
double first_image_time;
int pub_count = 1;
bool first_image_flag = true;
double last_image_time = 0;
bool init_pub = 0;

/*
 * brief: 
 * get image, operate detection and tracking, then pub the information in current image
 * 
 * in detail:
 * 1. if this is the first frame, set some values and return from this function
 * 2. if the time interval of the images is too big or the current time is smaller than last image time, reset the feature tracker, publish
 *    restart_flag and return from this function
 * 3. assign last_image_time using the time of img_msg in seconds
 * 4. if the pub_count is smaller than FREQ*time_interval, assign PUB_THIS_FRAME using true, when the time interval is rather big, reset
 *    pub_count to 0; else assign PUB_THIS_FRAME using false
 * 5. for every camera on the hardware platform
 *    5.1 if this is not the second camera or this is not the STEREO_TRACK mode, read the image, find corresponding points between two
 *        neighbouring frames(alse detect new features for the new image), and compute velocity in the undistorted camera coordinate
 *    5.2 else, this is the second camera in the STEREO_TRACK mode, just load the image
 * 6. update id for points
 * 7. if the macros tell that we should publish some information, publish corresponding points
 */
void img_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    /*
     * if this is the first image received
     * assign the bool value first_image_flag with false
     * assign the double value first_image_time using the time of img_msg in seconds
     * assign the double value last_image_time using the time of img_msg in seconds
     * return from this function
     */
    if(first_image_flag)
    {
        first_image_flag = false;
        first_image_time = img_msg->header.stamp.toSec();
        last_image_time = img_msg->header.stamp.toSec();
        return;
    }
    // detect unstable camera stream
    // if the time interval of the images is too big or the current time is smaller than last image time, reset the feature tracker, publish
    // restart_flag and return from this function
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
    // assign last_image_time using the time of img_msg in seconds
    last_image_time = img_msg->header.stamp.toSec();
    // frequency control

    // if the pub_count is relatively small, we should pub this frame so that the pub_count will increase
    if (round(1.0 * pub_count / (img_msg->header.stamp.toSec() - first_image_time)) <= FREQ)
    {
        PUB_THIS_FRAME = true;
        /* 
         * reset the frequency control
         * this condition will be satisfied when delta_time is rather big, if the delta_time is small, the fractional part of (pub_count/delta_time)
         * will not close to 0(here, I mean (0.99, 1] or [0, 0.01) ), so, I think the condition in the if means that lots of time passed since
         * the first_image_time, so we need to rest first_image_time tobe the time of img_msg in seconds and reset pub_count to be 0
         */ 
        if (abs(1.0 * pub_count / (img_msg->header.stamp.toSec() - first_image_time) - FREQ) < 0.01 * FREQ)
        {
            first_image_time = img_msg->header.stamp.toSec();
            pub_count = 0;
        }
    }
    // if the pub_count is not relatively small, we do not need to publish this frame
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
    /*
     * for every camera on the hardware platform
     */
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ROS_DEBUG("processing camera %d", i);
        /*
         * if this is not the second camera or this is not the STEREO_TRACK mode, read the image, find corresponding points between two adjacent frames(also 
         * detect new features for the new image), and compute velocity in the undistorted camera coordinate
         */
        if (i != 1 || !STEREO_TRACK)// the feature points was extracted and track here
            trackerData[i].readImage(ptr->image.rowRange(ROW * i, ROW * (i + 1)), img_msg->header.stamp.toSec());
        /*
         * I think corresponding to the if condition, this branch means that this is the second camera in the STEREO_TRACK mode
         * if EQUALIZE, applay contrast limited adaptive histogram equalization for the image
         * else, just load the image
         * we do not detect and track features of the second camera image in the STEREO_TRACK mode 
         */ 
        else
        {
            if (EQUALIZE)
            {
                // CLAHE contrast limited adaptive histogram equalization
                cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
                clahe->apply(ptr->image.rowRange(ROW * i, ROW * (i + 1)), trackerData[i].cur_img);
            }
            else
                trackerData[i].cur_img = ptr->image.rowRange(ROW * i, ROW * (i + 1));
        }

#if SHOW_UNDISTORTION
        // undistort the image and show the undistorted image with its name as the input of this function
        trackerData[i].showUndistortion("undistrotion_" + std::to_string(i));
#endif
    }

    /*
     * for integers from [0, ids.size() ), update id for ids[i] if one of the camera on the platform(not the second camera or not STEREO_TRACK mode), if none of 
     * cameras on the platform satisfing the condition not the second camera or not STEREO_TRACK mode observed the kth feature, break from the circulation and do not
     * process the following features [k, ids.size() )
     */
    for (unsigned int i = 0;; i++)
    {
        bool completed = false;
        for (int j = 0; j < NUM_OF_CAM; j++)
            if (j != 1 || !STEREO_TRACK)
                completed |= trackerData[j].updateID(i);//updateID will update id for a point, if the input i is no less than ids.size, return false
        if (!completed)
            break;
    }

   /*
    * if PUB_THIS_FRAME is true
    * 1. add pub_count by 1
    * 2. define a vector of set<int> hash_ids, the size of the vector is NUM_OF_CAM, each element in the vector stores point index of 
    *    trackerData[i]
    * 3. for all cameras on the hardware platform
    *    3.1 for every element in ids of trackerData[i], if the track_cnt of the j th point in trackerData[i] is bigger than 1, push the 
    *        id, undistorted coordinate, original pixel coordinate and velocity to corresponding containers
    * 4. if init_pub = false, assign init_pub using true, else, publish feature_points
    * 5. if SHOW_TRACK, draw feature points on the image and publish the image
    */
   if (PUB_THIS_FRAME)
   {
        pub_count++;
        sensor_msgs::PointCloudPtr feature_points(new sensor_msgs::PointCloud);
        /*
         * sensor_msgs::ChannelFloat32 has 2 parts: 1. string name    2.float32[] values
         */
        sensor_msgs::ChannelFloat32 id_of_point;
        sensor_msgs::ChannelFloat32 u_of_point;
        sensor_msgs::ChannelFloat32 v_of_point;
        sensor_msgs::ChannelFloat32 velocity_x_of_point;
        sensor_msgs::ChannelFloat32 velocity_y_of_point;

        feature_points->header = img_msg->header;
        feature_points->header.frame_id = "world";

        vector<set<int>> hash_ids(NUM_OF_CAM);// vector of set<int>, the size of the vector is NUM_OF_CAM
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            // the size of the array trackerData is NUM_OF_CAM
            auto &un_pts = trackerData[i].cur_un_pts;
            auto &cur_pts = trackerData[i].cur_pts;
            auto &ids = trackerData[i].ids;
            auto &pts_velocity = trackerData[i].pts_velocity;
            // for every element in ids of trackerData[i]
            for (unsigned int j = 0; j < ids.size(); j++)
            {
                /*
                 * if the track_cnt of the j th point in trackerData[i] is bigger than 1
                 * push the id, undistorted coordinate, original pixel coordinate and velocity to corresponding containers
                 */
                if (trackerData[i].track_cnt[j] > 1)// vector<int> track_cnt
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
        ROS_DEBUG("publish %f, at %f", feature_points->header.stamp.toSec(), ros::Time::now().toSec());
        // skip the first image; since no optical speed on frist image
        if (!init_pub)
        {
            init_pub = 1;
        }
        // feature_points is the pointcloud representing points on the plane z=1
        else
            pub_img.publish(feature_points);

        /*
         * if SHOW_TRACK
         *    1. for every camera on the hardware platform
         *       1.1 for every point in cur_pts of trackerData[i], draw a circle in the image centered at the point
         *    2. publish the image
         */
        if (SHOW_TRACK)
        {
            ptr = cv_bridge::cvtColor(ptr, sensor_msgs::image_encodings::BGR8);
            //cv::Mat stereo_img(ROW * NUM_OF_CAM, COL, CV_8UC3);
            cv::Mat stereo_img = ptr->image;

            for (int i = 0; i < NUM_OF_CAM; i++)// for every camera
            {
                cv::Mat tmp_img = stereo_img.rowRange(i * ROW, (i + 1) * ROW);
                cv::cvtColor(show_img, tmp_img, CV_GRAY2RGB);

                // for every point in cur_pts of trackerData[i], draw a circle in the image centered at the point
                for (unsigned int j = 0; j < trackerData[i].cur_pts.size(); j++)
                {
                    double len = std::min(1.0, 1.0 * trackerData[i].track_cnt[j] / WINDOW_SIZE);
                    // draw feature points on image
                    cv::circle(tmp_img, trackerData[i].cur_pts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
                    //draw speed line
                    /*
                    Vector2d tmp_cur_un_pts (trackerData[i].cur_un_pts[j].x, trackerData[i].cur_un_pts[j].y);
                    Vector2d tmp_pts_velocity (trackerData[i].pts_velocity[j].x, trackerData[i].pts_velocity[j].y);
                    Vector3d tmp_prev_un_pts;
                    tmp_prev_un_pts.head(2) = tmp_cur_un_pts - 0.10 * tmp_pts_velocity;
                    tmp_prev_un_pts.z() = 1;
                    Vector2d tmp_prev_uv;
                    trackerData[i].m_camera->spaceToPlane(tmp_prev_un_pts, tmp_prev_uv);
                    cv::line(tmp_img, trackerData[i].cur_pts[j], cv::Point2f(tmp_prev_uv.x(), tmp_prev_uv.y()), cv::Scalar(255 , 0, 0), 1 , 8, 0);
                    */
                    //char name[10];
                    //sprintf(name, "%d", trackerData[i].ids[j]);
                    //cv::putText(tmp_img, name, trackerData[i].cur_pts[j], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
                }
            }
            //cv::imshow("vis", stereo_img);
            //cv::waitKey(5);
            pub_match.publish(ptr->toImageMsg());
        }
    }
    ROS_INFO("whole feature tracker processing costs: %f", t_r.toc());
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "feature_tracker");
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

    ros::Subscriber sub_img = n.subscribe(IMAGE_TOPIC, 100, img_callback);

    pub_img = n.advertise<sensor_msgs::PointCloud>("feature", 1000); // publish feature points as point cloud
    pub_match = n.advertise<sensor_msgs::Image>("feature_img",1000); // if SHOW_TRACK is true, publish image with feature points draw on it
    // if the time interval of the images is too big or the current time is smaller than last image time, publish restart flag
    pub_restart = n.advertise<std_msgs::Bool>("restart",1000);
    /*
    if (SHOW_TRACK)
        cv::namedWindow("vis", cv::WINDOW_NORMAL);
    */
    ros::spin();
    return 0;
}


// new points velocity is 0, pub or not?
// track cnt > 1 pub?