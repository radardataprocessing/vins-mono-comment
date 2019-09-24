#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include "estimator.h"
#include "parameters.h"
#include "utility/visualization.h"


Estimator estimator;

std::condition_variable con;
double current_time = -1;
queue<sensor_msgs::ImuConstPtr> imu_buf;
queue<sensor_msgs::PointCloudConstPtr> feature_buf;
queue<sensor_msgs::PointCloudConstPtr> relo_buf;
int sum_of_wait = 0;

std::mutex m_buf;
std::mutex m_state;
std::mutex i_buf;
std::mutex m_estimator;

double latest_time;
Eigen::Vector3d tmp_P;
Eigen::Quaterniond tmp_Q;
Eigen::Vector3d tmp_V;
Eigen::Vector3d tmp_Ba;
Eigen::Vector3d tmp_Bg;
Eigen::Vector3d acc_0;
Eigen::Vector3d gyr_0;
bool init_feature = 0;
bool init_imu = 1;
double last_imu_t = 0;

/*
 * 1. get the time of the imu message in the parameter list as t, use the tmp_Q, tmp_Ba and estimator.g as well as the acceleration measurement of last time to compute 
 *    the real acceleration of last time, use tmp_Bg and the measurement of gyro last time and this time to compute the average angular velocity of the adjacent two frames,
 *    use the tmp_Q, tmp_ba, estimator.g and the acceleration of this time to compute the real acceleration of this time, compute the average acceleration of the adjacent
 *    two frames.
 * 2. compute new state using qt=q0*deltaQ(un_gyr*dt), pt=p0+v*t+1/2*a*t*t, vt=v0+a*t and set acc_0 and gyr_0 to be the acceleration and gyro measurement of this time
 */
void predict(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = imu_msg->header.stamp.toSec();
    if (init_imu)// if this is the first imu msg, set the latest_time tobe the time of this msg
    {
        latest_time = t;
        init_imu = 0;
        return;
    }
    double dt = t - latest_time;
    latest_time = t;

    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    Eigen::Vector3d linear_acceleration{dx, dy, dz};

    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Eigen::Vector3d angular_velocity{rx, ry, rz};

    Eigen::Vector3d un_acc_0 = tmp_Q * (acc_0 - tmp_Ba) - estimator.g;// the real acceleration of the last time
    // acc_0 - tmp_Ba eliminate the influence of the bias
    // tmp_Q * (acc_0 - tmp_Ba) eliminate the influence of the orientation in the world coordinate
    // tmp_Q * (acc_0 - tmp_Ba) - estimator.g eliminate the influence of gravity
    // after the above 3 steps, the un_acc_0 is the acceleration that actually influence the motion

    Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - tmp_Bg;// the average angular velocity of the adjacent two time stamp
    tmp_Q = tmp_Q * Utility::deltaQ(un_gyr * dt);

    Eigen::Vector3d un_acc_1 = tmp_Q * (linear_acceleration - tmp_Ba) - estimator.g;// the real acceleration of this time

    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);// the average acceleration of the adjacent two timestamps

    tmp_P = tmp_P + dt * tmp_V + 0.5 * dt * dt * un_acc;// pt = p0 + v*t + 1/2*a*t*t
    tmp_V = tmp_V + dt * un_acc;// vt = v0 + a*t

    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

/*
 * for every message in the queue imu_buf, integrate the imu measurements into motion
 */
void update()//integrate the imu measurements into imu motion
{
    TicToc t_predict;
    latest_time = current_time;
    // I think the macro WINDOW_SIZE in the bracket means that we will keep WINDOW_SIZE+1 imu states in the estimator
    tmp_P = estimator.Ps[WINDOW_SIZE];// position
    tmp_Q = estimator.Rs[WINDOW_SIZE];// orientation
    tmp_V = estimator.Vs[WINDOW_SIZE];// velocity
    tmp_Ba = estimator.Bas[WINDOW_SIZE];// bias of acceleration
    tmp_Bg = estimator.Bgs[WINDOW_SIZE];// bias of gyro
    acc_0 = estimator.acc_0;
    gyr_0 = estimator.gyr_0;

    queue<sensor_msgs::ImuConstPtr> tmp_imu_buf = imu_buf;
    for (sensor_msgs::ImuConstPtr tmp_imu_msg; !tmp_imu_buf.empty(); tmp_imu_buf.pop())
        predict(tmp_imu_buf.front());

}

/*
 * 1. if the imu_buf or the feature_buf is empty, return vector measurements
 * 2. if the biggest imu time in the queue is not bigger than the smallest feature time plus td, add sum_of_wait by 1 and return measurements
 * 3. if the smallest imu time in the queue is not smaller than the smallest feature time plus td, throw the image
 * 4. get the front of the feature_buf as img_msg and pop the message from feature_buf
 * 5. while the smallest time in imu_buf is smaller than the img_msg time plus td, put the imu_buf front to std::vector<sensor_msgs::ImuConstPtr> IMUS
 * 6. emplace pair of imu vector and image message to the vector 
 * 7. return measurements
 */
std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>>
getMeasurements()// this function returns vector of pair of imu msg vector and point cloud msg
{
    std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;

    while (true)
    {
        if (imu_buf.empty() || feature_buf.empty())// if the imu buf or the feature buf is empty, return vector measurements
            return measurements;

        // if the biggest imu time in the queue is not bigger than the smallest feature time plus td, add sum_of_wait by 1 and return measurements
        if (!(imu_buf.back()->header.stamp.toSec() > feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            //ROS_WARN("wait for imu, only should happen at the beginning");
            sum_of_wait++;
            return measurements;
        }

        // if the smallest imu time in the queue is not smaller than the smallest feature time plus td, throw image
        if (!(imu_buf.front()->header.stamp.toSec() < feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            ROS_WARN("throw img, only should happen at the beginning");
            feature_buf.pop();
            continue;
        }
        sensor_msgs::PointCloudConstPtr img_msg = feature_buf.front();// get the front of the feature_buf as the img_msg
        feature_buf.pop();

        std::vector<sensor_msgs::ImuConstPtr> IMUs;
        // while the smallest time in imu_buf is smaller than the img_msg time plus td, put the imu_buf front to IMUS
        while (imu_buf.front()->header.stamp.toSec() < img_msg->header.stamp.toSec() + estimator.td)
        {
            IMUs.emplace_back(imu_buf.front());
            imu_buf.pop();
        }
        IMUs.emplace_back(imu_buf.front());
        if (IMUs.empty())
            ROS_WARN("no imu between two image");
        measurements.emplace_back(IMUs, img_msg);//emplace pair of imus vector and image message to the vector
    }
    return measurements;
}

/*
 * 1. if the imu time is smaller than the last imu time, warn that imu message is in wrong order and return from the function
 * 2. set the timestamp of this imu message as last_imu_t
 * 3. use the imu_msg to predict tmp_P, tmp_Q and tmp_V
 * 4. if the solver_flag of the estimator is NON_LINEAR, publish odometry composed of tmp_P, tmp_Q and tmp_V
 */
void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    if (imu_msg->header.stamp.toSec() <= last_imu_t)// if the imu time is smaller than the last imu time
    {
        ROS_WARN("imu message in disorder!");
        return;
    }
    last_imu_t = imu_msg->header.stamp.toSec();

    m_buf.lock();
    imu_buf.push(imu_msg);
    m_buf.unlock();
    con.notify_one();

    last_imu_t = imu_msg->header.stamp.toSec();

    {
        std::lock_guard<std::mutex> lg(m_state);
        predict(imu_msg);
        std_msgs::Header header = imu_msg->header;
        header.frame_id = "world";
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            pubLatestOdometry(tmp_P, tmp_Q, tmp_V, header);
            // construct a nav_msgs::Odometry message, set the position tobe odometry.pose.pose.position, set the orientation tobe
            // odometry.pose.pose.orientation, set the velocity tobe odometry.twist.twist.linear
    }
}

/*
 * 1. if this is the first detected feature, skip it because it doesn't contain optical flow speed, just return from this function
 * 2. push the feature_msg into feature_buf
 */
void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
    // skip the first frame of feature, it doesn't contain optical flow speed
    if (!init_feature)
    {
        //skip the first detected feature, which doesn't contain optical flow speed
        init_feature = 1;
        return;
    }
    m_buf.lock();
    feature_buf.push(feature_msg);
    m_buf.unlock();
    con.notify_one();
}

/*
 * if receive a restart signal, clear the buf and estimator
 */
void restart_callback(const std_msgs::BoolConstPtr &restart_msg)
{
    if (restart_msg->data == true)
    {
        ROS_WARN("restart the estimator!");
        m_buf.lock();
        while(!feature_buf.empty())
            feature_buf.pop();
        while(!imu_buf.empty())
            imu_buf.pop();
        m_buf.unlock();
        m_estimator.lock();
        estimator.clearState();
        estimator.setParameter();
        m_estimator.unlock();
        current_time = -1;
        last_imu_t = 0;
    }
    return;
}

// if received the pointcloud message, push the message into relo_buf
void relocalization_callback(const sensor_msgs::PointCloudConstPtr &points_msg)
{
    //printf("relocalization callback! \n");
    m_buf.lock();
    relo_buf.push(points_msg);
    m_buf.unlock();
}

// thread: visual-inertial odometry
/*
 * measurements is a vector, its element is a pair of imu_msg vector and pointcloud msg
 * 1. for every measurement in the vector measurements
 *    1.1 for every imu_msg in the pair measurement, define t as the stamp of imu_msg, define img_t as the stamp of img_msg plus estimator.td
 *        1.1.1 if t is no more than img_t, integrate the imu measurements to position, orientation and velocity
 *        1.1.2 if t is more than img_t, interpolationthe two imu_msg around the img msg as a new imu msg, integrate the new imu msg to position,
 *              orientation and velocity   
 *    1.2 if the relo_buf is not empty, get the latest(last one in the queue) msg in relo_buf as relo_msg, if the relo_msg is not null, get all
 *        points in relo_msg and push them into match_points, get the frame_index and pose of the relo_msg, set the relo_frame for the estimator
 *        using match_points and frame_index and pose of the reloc frame
 *    1.3 for the points in the img_msg in the pair measurement, get the point's xyz_uv_velocity, camera_id and feature_id, put the pair of 
 *        camera_id and xyz_uv_velocity as the value of the feature_id key
 */
void process()
{
    while (true)
    {
        // measurements is a vector , whose element is a pair of imu_msg_vector and pointcloud msg
        std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;
        std::unique_lock<std::mutex> lk(m_buf);
        con.wait(lk, [&]
                 {
            return (measurements = getMeasurements()).size() != 0;
                 });
        lk.unlock();
        m_estimator.lock();
        for (auto &measurement : measurements)// for every measurement in the vector measurements
        {
            auto img_msg = measurement.second;
            double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
            for (auto &imu_msg : measurement.first)// for every imu msg in an element of measurements
            {
                double t = imu_msg->header.stamp.toSec();
                double img_t = img_msg->header.stamp.toSec() + estimator.td;// compute the real time of image as img_t=img_msg.time+td
                // this if switch means whethre the imu time tobe processed is smaller than the image time, if it is smaller than the img time, just process it;
                // if it is bigger than the image time, bilinear intepolate the last imu data and current imu data, then process the intepolated data
                if (t <= img_t)// if the imu msg time is smaller than the real image time
                { 
                    if (current_time < 0)
                        current_time = t;
                    double dt = t - current_time;
                    ROS_ASSERT(dt >= 0);
                    current_time = t;
                    dx = imu_msg->linear_acceleration.x;
                    dy = imu_msg->linear_acceleration.y;
                    dz = imu_msg->linear_acceleration.z;
                    rx = imu_msg->angular_velocity.x;
                    ry = imu_msg->angular_velocity.y;
                    rz = imu_msg->angular_velocity.z;
                    estimator.processIMU(dt, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    //printf("imu: dt:%f a: %f %f %f w: %f %f %f\n",dt, dx, dy, dz, rx, ry, rz);

                }
                else
                {
                    double dt_1 = img_t - current_time;// time interval between the img time and old imu time
                    double dt_2 = t - img_t;// time interval between the new imu time and img time
                    current_time = img_t;
                    ROS_ASSERT(dt_1 >= 0);
                    ROS_ASSERT(dt_2 >= 0);
                    ROS_ASSERT(dt_1 + dt_2 > 0);
                    double w1 = dt_2 / (dt_1 + dt_2);
                    double w2 = dt_1 / (dt_1 + dt_2);
                    dx = w1 * dx + w2 * imu_msg->linear_acceleration.x;
                    dy = w1 * dy + w2 * imu_msg->linear_acceleration.y;
                    dz = w1 * dz + w2 * imu_msg->linear_acceleration.z;
                    rx = w1 * rx + w2 * imu_msg->angular_velocity.x;
                    ry = w1 * ry + w2 * imu_msg->angular_velocity.y;
                    rz = w1 * rz + w2 * imu_msg->angular_velocity.z;
                    // interpolate the imu value using two time intervals by bilinear interpolation, the time stamp of the interpolated imu value is img time
                    estimator.processIMU(dt_1, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    //printf("dimu: dt:%f a: %f %f %f w: %f %f %f\n",dt_1, dx, dy, dz, rx, ry, rz);
                }
            }
            // set relocalization frame
            sensor_msgs::PointCloudConstPtr relo_msg = NULL;
            while (!relo_buf.empty())// if the relo_buf is not null, get the latest relo_msg
            {
                relo_msg = relo_buf.front();
                relo_buf.pop();
            }
            if (relo_msg != NULL)// if here actually exsists pointcloud relo msg
            {
                vector<Vector3d> match_points;
                double frame_stamp = relo_msg->header.stamp.toSec();
                for (unsigned int i = 0; i < relo_msg->points.size(); i++)// push all 3D points in relo_msg into vector match_points
                {
                    Vector3d u_v_id;
                    u_v_id.x() = relo_msg->points[i].x;
                    u_v_id.y() = relo_msg->points[i].y;
                    u_v_id.z() = relo_msg->points[i].z;
                    match_points.push_back(u_v_id);
                }
                // get the pose of the reloc frame
                Vector3d relo_t(relo_msg->channels[0].values[0], relo_msg->channels[0].values[1], relo_msg->channels[0].values[2]);
                Quaterniond relo_q(relo_msg->channels[0].values[3], relo_msg->channels[0].values[4], relo_msg->channels[0].values[5], relo_msg->channels[0].values[6]);
                Matrix3d relo_r = relo_q.toRotationMatrix();
                int frame_index;// get the frame index of the reloc frame
                frame_index = relo_msg->channels[0].values[7];
                estimator.setReloFrame(frame_stamp, frame_index, match_points, relo_t, relo_r);// set reloc frame for the estimator
            }

            ROS_DEBUG("processing vision data with stamp %f \n", img_msg->header.stamp.toSec());

            TicToc t_s;
            //此处保存的是特征的id,及其对应特征点的相机id,三维坐标，二维像素坐标以及像素坐标速度
            map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> image;
            // image is a map of a int and a vector, the int value stores the feature id, the vector stores a series of pairs, each pair store the camera id and 
            // the feature point's 3D position, 2D coordinate and velocity, here the camera id means the camera id in case of multi-camera
            for (unsigned int i = 0; i < img_msg->points.size(); i++)
            {
                int v = img_msg->channels[0].values[i] + 0.5;
                // in the process of constructing the image_msg, the id_of_point is pushed into channels[0], id_of_point=p_id*NUM_OF_CAM+camera_id
                int feature_id = v / NUM_OF_CAM;
                int camera_id = v % NUM_OF_CAM;
                double x = img_msg->points[i].x;
                double y = img_msg->points[i].y;
                double z = img_msg->points[i].z;
                double p_u = img_msg->channels[1].values[i];
                double p_v = img_msg->channels[2].values[i];
                double velocity_x = img_msg->channels[3].values[i];
                double velocity_y = img_msg->channels[4].values[i];
                ROS_ASSERT(z == 1);
                Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
                xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
                image[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
            }
            estimator.processImage(image, img_msg->header);

            double whole_t = t_s.toc();
            printStatistics(estimator, whole_t);
            std_msgs::Header header = img_msg->header;
            header.frame_id = "world";

            pubOdometry(estimator, header);
            pubKeyPoses(estimator, header);
            pubCameraPose(estimator, header);
            pubPointCloud(estimator, header);
            pubTF(estimator, header);
            pubKeyframe(estimator);
            if (relo_msg != NULL)
                pubRelocalization(estimator);
            //ROS_ERROR("end: %f, at %f", img_msg->header.stamp.toSec(), ros::Time::now().toSec());
        }
        m_estimator.unlock();
        m_buf.lock();
        m_state.lock();
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            update();
        m_state.unlock();
        m_buf.unlock();
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "vins_estimator");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    readParameters(n);
    estimator.setParameter();
#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif
    ROS_WARN("waiting for image and imu...");

    registerPub(n);

    ros::Subscriber sub_imu = n.subscribe(IMU_TOPIC, 2000, imu_callback, ros::TransportHints().tcpNoDelay());
    ros::Subscriber sub_image = n.subscribe("/feature_tracker/feature", 2000, feature_callback);
    ros::Subscriber sub_restart = n.subscribe("/feature_tracker/restart", 2000, restart_callback);
    ros::Subscriber sub_relo_points = n.subscribe("/pose_graph/match_points", 2000, relocalization_callback);

    std::thread measurement_process{process};
    ros::spin();

    return 0;
}
