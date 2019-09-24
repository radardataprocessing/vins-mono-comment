#include "estimator.h"

Estimator::Estimator(): f_manager{Rs}
{
    ROS_INFO("init begins");
    clearState();
}

void Estimator::setParameter()
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = TIC[i];
        ric[i] = RIC[i];
    }
    f_manager.setRic(ric);
    ProjectionFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionTdFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    td = TD;
}

/*
 * 1. clear Rs, Ps, Vs, Bas, Bgs, dt_buf, linear_acceleration_buf, angular_velocity_buf and pre_integrations in the window
 * 2. for every cam of the system, clear tic and ric
 * 3. set solver_flag=INITIAL, first_imu=false, sum_of_back=0, sum_of_front=0, frame_count=0, initial_timestamp=0, td=TD; and clear all_image_frame
 * 4. set tmp_pre_integration and last_marginalization_info as nullptr, clear last_marginalization_parameter_blocks
 * 5. clear state of f_manager
 * 6. set failure_occur=0, relocalization_info=0
 * 7. set drift_correct_r and drift_correct_t as 0
 */
void Estimator::clearState()
{
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();
        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();

        if (pre_integrations[i] != nullptr)
        {
            delete pre_integrations[i];
        }
        pre_integrations[i] = nullptr;
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d::Zero();
        ric[i] = Matrix3d::Identity();
    }

    solver_flag = INITIAL;
    first_imu = false,
    sum_of_back = 0;
    sum_of_front = 0;
    frame_count = 0;
    solver_flag = INITIAL;
    initial_timestamp = 0;
    all_image_frame.clear(); // map<double, ImageFrame> all_image_frame;
    td = TD;
    /* 
     * ImageFrame is a class representing an image frame who contain 
     * 1. a map representing all features in the frame 
     * 2. a double type value t
     * 3. a Matrix3d R, I think it is the result of visual computation
     * 4. a Vector3d T  
     * 5. a pointer pointing to IntegrationBase object, I think this is the result of imu computation
     * 6. a bool type value representing whether this is a keyframe
     */


    if (tmp_pre_integration != nullptr)
        delete tmp_pre_integration;
    if (last_marginalization_info != nullptr)
        delete last_marginalization_info;

    tmp_pre_integration = nullptr;
    last_marginalization_info = nullptr;
    last_marginalization_parameter_blocks.clear();

    f_manager.clearState();

    failure_occur = 0;
    relocalization_info = 0;

    drift_correct_r = Matrix3d::Identity();
    drift_correct_t = Vector3d::Zero();
}

/*
 * 1. if this is the first imu message, set the acc and gyro value in the parameter list as acc_0 and gyro_0
 * 2. if the (frame_count)th pre_integrations is NULL, alloc an IntegrationBase object
 * 3. if the frame_count is not 0, push the dt, linear_acceleration and angular_velocity into dt_buf, acc_buf and gyro_buf of the class IntegrationBase's
 *    object pre_integrations[frame_count] and tmp_pre_integration, use linear_acceleration anf angular_velocity to predict Rs, Ps and Vs
 * 4. set acc_0 to be linear_acceleration and set gyr_0 to be angular_velocity
 */
void Estimator::processIMU(double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity)
{
    if (!first_imu)// if this is the first imu message, set the acce and gyro value in the parameter list as acc_0 and gyro_0
    {
        first_imu = true;
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
    }

    if (!pre_integrations[frame_count])  // IntegrationBase *pre_integrations[(WINDOW_SIZE + 1)]
    {
        // IntegrationBase *pre_integrations[(WINDOW_SIZE + 1)];
        pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
    }
    if (frame_count != 0)
    {
        // push_back functino here push the dt, linear_acceleration and angular_velocity into the dt_buf, acc_buf and gyr_buf of the class IntegrationBase
        // the pre integration result of the imu data is stored in the class IntegrationBase as delta_p, delta_q, delta_v, linearized_ba, linearized_bg
        pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);
        //if(solver_flag != NON_LINEAR)
            tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);

        dt_buf[frame_count].push_back(dt);
        linear_acceleration_buf[frame_count].push_back(linear_acceleration);
        angular_velocity_buf[frame_count].push_back(angular_velocity);

        int j = frame_count;         
        Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;
        Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
        Vs[j] += dt * un_acc;
    }
    acc_0 = linear_acceleration;// after current data processed, set last data tobe current data
    gyr_0 = angular_velocity;
}

/*
 * image is a map of a int and a vector, the int value stores the feature id, the vector stores a series of pairs, each pair store the camera id and the 
 * feature points's 3D position, 2D coordinate and velocity, here the camera id means the camera id in case of multi-camera
 * 1. if the current frame has enough parallax with the previous frame, marginalize the second new frame, else, it means the second new frame is not
 *    signaficant, marginalize the second new frame. If marginalize second new frame, it means this frame is rejected and is a non_keyframe
 */
void Estimator::processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const std_msgs::Header &header)
{
    // image is a map of a int and a vector, the int value stores the feature id, the vector stores a series of pairs, each pair store the camera id and 
    // the feature point's 3D position, 2D coordinate and velocity, here the camera id means the camera id in case of multi-camera
    ROS_DEBUG("new image coming ------------------------------------------");
    ROS_DEBUG("Adding feature points %lu", image.size());
    // if the current frame has small parallax with the prev frame, marginalize the second new frame; else marginalize the old frame
    // if the current frame has enough parallax with the previous frames, marginalize the old frame; else way, it means the second new frame is not
    // signaficant, marginalize the second new frame 
    if (f_manager.addFeatureCheckParallax(frame_count, image, td))
        marginalization_flag = MARGIN_OLD;// for example the drone is moving very fast   MARGIN_OLD=0
    else
        marginalization_flag = MARGIN_SECOND_NEW;// for example the drone is hovering    MARGIN_SECOND_NEW=1

    /*
    enum MarginalizationFlag
    {
        MARGIN_OLD = 0,
        MARGIN_SECOND_NEW = 1
    };
    */

    // if marginalize second new frame, it means this frame is rejected, and non-keyframe
    ROS_DEBUG("this frame is--------------------%s", marginalization_flag ? "reject" : "accept");
    ROS_DEBUG("%s", marginalization_flag ? "Non-keyframe" : "Keyframe");
    ROS_DEBUG("Solving %d", frame_count);
    ROS_DEBUG("number of feature: %d", f_manager.getFeatureCount());// get the number of valid features
    // Headers is an array of std_msgs whose size is WINDOW_SIZE+1
    Headers[frame_count] = header;

    ImageFrame imageframe(image, header.stamp.toSec());
    imageframe.pre_integration = tmp_pre_integration;
    all_image_frame.insert(make_pair(header.stamp.toSec(), imageframe));
    tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};

    /*
     * ESTIMATE_EXTRINSIC == 0 means the extrinsic param is fixed
     * ESTIMATE_EXTRINSIC == 1 means the initial guess of extrinsic is given, so we can optimize around the initial guess
     * ESTIMATE_EXTRINSIC == 2 means no prior about the extrinsic is given, we need to calibrate the extrinsic
     */
    if(ESTIMATE_EXTRINSIC == 2)
    {
        ROS_INFO("calibrating extrinsic param, rotation movement is needed");
        if (frame_count != 0)
        {
            // go through all the feature in list feature to get all correspondings for the two frames frame_count-1 and frame_count
            vector<pair<Vector3d, Vector3d>> corres = f_manager.getCorresponding(frame_count - 1, frame_count);
            Matrix3d calib_ric;
            // compute the rotation matrix of imu and camera then write the result into calib_ric
            if (initial_ex_rotation.CalibrationExRotation(corres, pre_integrations[frame_count]->delta_q, calib_ric))
            {
                ROS_WARN("initial extrinsic rotation calib success");
                ROS_WARN_STREAM("initial extrinsic rotation: " << endl << calib_ric);
                ric[0] = calib_ric;
                RIC[0] = calib_ric;
                ESTIMATE_EXTRINSIC = 1;// after CalibrationExRotation, change ESTIMATE_EXTRINSIC from 2 to 1
            }
        }
    }

    if (solver_flag == INITIAL)// if the solver_flag is INITIAL
    {
        if (frame_count == WINDOW_SIZE)// if the frame_count equals WINDOW_SIZE
        {
            bool result = false;
            // if the extrinsic is already initialed or don't need to be estimate, and the header time is 0.1 bigger than initial_timestamp 
            if( ESTIMATE_EXTRINSIC != 2 && (header.stamp.toSec() - initial_timestamp) > 0.1)
            {
               result = initialStructure();
               initial_timestamp = header.stamp.toSec();
            }
            /*
             * enum SolverFlag
             * {
             *    INITIAL,
             *    NON_LINEAR
             * }
             */
            if(result)// if initialStructure succeed
            {
                solver_flag = NON_LINEAR;
                solveOdometry();
                slideWindow();
                f_manager.removeFailures();
                ROS_INFO("Initialization finish!");
                last_R = Rs[WINDOW_SIZE];
                last_P = Ps[WINDOW_SIZE];
                last_R0 = Rs[0];
                last_P0 = Ps[0];
                
            }
            else
                slideWindow();
        }
        else// if the frame_count does not equal WINDOW_SIZE, add frame_count by 1
            frame_count++;
    }
    else
    {
        TicToc t_solve;
        solveOdometry();
        ROS_DEBUG("solver costs: %fms", t_solve.toc());

        if (failureDetection())
        {
            ROS_WARN("failure detection!");
            failure_occur = 1;
            clearState();
            setParameter();
            ROS_WARN("system reboot!");
            return;
        }

        TicToc t_margin;
        slideWindow();
        f_manager.removeFailures();
        ROS_DEBUG("marginalization costs: %fms", t_margin.toc());
        // prepare output of VINS
        key_poses.clear();
        for (int i = 0; i <= WINDOW_SIZE; i++)
            key_poses.push_back(Ps[i]);

        last_R = Rs[WINDOW_SIZE];
        last_P = Ps[WINDOW_SIZE];
        last_R0 = Rs[0];
        last_P0 = Ps[0];
    }
}

/*
 * 1. check imu observibility, compute the average of all image frame's acceleration as g, use the average and every acceleration to compute the
 *    variance, if the variance is too small, print info that the imu excitation is not enough
 * 2. for every feature in the f_manager, push the map of index and 2D points into the observation of tmp_feature, and push the tmp_feature of every 
 *    feature into sfm_f
 * 3. get relativePose
 *    3.1 for every frame in range [0, WINDOW_SIZE)
 *    3.1.1 go through all the feature in list feature to get all correspondings for frame i and frame WINDOW_SIZE
 *    3.1.2 if the number of correspondings of the two frames is bigger than 20
 *        3.1.2.1 compute the average disparity of all the correspondings
 *        3.1.2.2 if the average disparity is bigger than threshold and we succeed to solve relative r t according to the correspondings, set l to
 *              be r and return true
 *    3.2 if none of the frame in the window can successfully compute R and T with frame WINDOW_SIZE, return false
 *    if we can not solve pose using the frames in window, tell the user to move device around and return false
 * 4. first, use frame l as an fundamental to solve pnp to get poses of other frames and triangulate feature points, then use certain frames
 *    to triangulate feature points, then construct an optimization problem to optimize frame's pose and feature position
 *    if the pnp or optimization problem didn't been solved successfully, return false
 * 5. for all frames in all_image_frame
 *    5.1 if the frame is in the window, set is_key_frame of the ImageFrame tobe true, and set R, T for the ImageFrame, add i by 1 and continue
 *        to process next frame
 *    5.2 if the time of the frame is bigger than the ith time in window, add i by 1
 *    5.3 use the ith Q and T to get R_initial and P_initial for the current frame(even though this frame is not the key frame)
 *    5.4 for every feature in the frame, if we can find the feature in the map sfm_tracked_points, push the world coordinate of the feature point 
 *        into pts_3_vector, push the image pixel coordinate of the frame into pts_2_vector
 *    5.5 if we can only find less than 6 feature points from the structure from motion result, print the number and print the points 
 *        is not enough for solve pnp and return false  
 *    5.6 apply opencv solvepnp for pts_3_vector and pts_2_vector, if we can not solve pnp using the corresponding world coordinate and pixel 
 *        coordinate,print solve pnp failed and return false
 *    5.7 use the pnp result to reset pose for frame_it
 * 6. if visualInitialAlign() returns true, return true, else print misalign visual structure with imu and return false
 */
bool Estimator::initialStructure()
{
    TicToc t_sfm;
    //check imu observibility
    {
        map<double, ImageFrame>::iterator frame_it;
        Vector3d sum_g;
        // add up all image frame's acceleration as the g, the first frame do not have any integration information, so just skip it
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            sum_g += tmp_g;
        }
        Vector3d aver_g;
        aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1);// compute the average of the above added g
        double var = 0;
        // use the average g and every acceleration to compute the variance
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
            //cout << "frame g " << tmp_g.transpose() << endl;
        }
        var = sqrt(var / ((int)all_image_frame.size() - 1));
        //ROS_WARN("IMU variation %f!", var);
        if(var < 0.25)// if the variance is too small, print info that the imu excitation is not enough
        {
            ROS_INFO("IMU excitation not enouth!");
            //return false;
        }
    }
    /*
     * struct SFMFeature
     * {
     *     bool state;
     *     int id;
     *     vector<pair<int,Vector2d>> observation;
     *     double position[3];
     *     double depth;
     * };
     */
    // global sfm
    Quaterniond Q[frame_count + 1];
    Vector3d T[frame_count + 1];
    map<int, Vector3d> sfm_tracked_points;
    vector<SFMFeature> sfm_f;
    // FeaturePerId contains a vector of FeaturePerFrame object
    for (auto &it_per_id : f_manager.feature) // list<FeaturePerId> feature   for every feature in the feature manager
    {
        int imu_j = it_per_id.start_frame - 1;
        SFMFeature tmp_feature;
        tmp_feature.state = false;
        tmp_feature.id = it_per_id.feature_id;
        for (auto &it_per_frame : it_per_id.feature_per_frame)// for all frames observing the certain feature
        {
            imu_j++;
            Vector3d pts_j = it_per_frame.point;
            // push the map of index and 2D points into the observation of tmp_feature
            tmp_feature.observation.push_back(make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
        }
        sfm_f.push_back(tmp_feature);
    } 
    Matrix3d relative_R;
    Vector3d relative_T;
    int l;
    /* relativePose
     * 1. for every frame in range [0, WINDOW_SIZE)
     *    1.1 go through all the feature in list feature to get all correspondings for frame i and frame WINDOW_SIZE
     *    1.2 if the number of correspondings of the two frames is bigger than 20
     *        1.2.1 compute the average disparity of all the correspondings
     *        1.2.2 if the average disparity is bigger than threshold and we succeed to solve relative r t according to the correspondings, set l to
     *              be r and return true
     * 2. if none of the frame in the window can successfully compute R and T with frame WINDOW_SIZE, return false
     */
    // if we can not solve pose using the frames in window, tell the user to move device around and return false
    if (!relativePose(relative_R, relative_T, l))
    {
        ROS_INFO("Not enough features or parallax; Move device around");
        return false;
    }
    GlobalSFM sfm;
    /*
     * first, use frame l as an fundamental to solve pnp to get poses of other frames and triangulate feature points, then use certain frames
     * to triangulate feature points, then construct an optimization problem to optimize frame's pose and feature position
     * if the pnp or optimization problem didn't been solved successfully, return false
     */
    // if the construct of the sfm did not suceed, print global SFM failed, set marginalization_flag to be MARGIN_OLD, and return false
    if(!sfm.construct(frame_count + 1, Q, T, l,
              relative_R, relative_T,
              sfm_f, sfm_tracked_points))
    {
        ROS_DEBUG("global SFM failed!");
        marginalization_flag = MARGIN_OLD;
        return false;
    }

    //solve pnp for all frame
    map<double, ImageFrame>::iterator frame_it;
    map<int, Vector3d>::iterator it;
    frame_it = all_image_frame.begin( );
    // i is the index for Q,T and Headers in window
    for (int i = 0; frame_it != all_image_frame.end( ); frame_it++)
    {
        // provide initial guess
        cv::Mat r, rvec, t, D, tmp_r;
        // Headers is an array of std_msgs whose size is WINDOW_SIZE+1
        // if the frame is in the window, set is_key_frame of the ImageFrame tobe true, and set R, T for the ImageFrame, add i by 1 and continue
        // to process next frame
        if((frame_it->first) == Headers[i].stamp.toSec())// if the time of the frame equals the ith time in the array Headers
        {
            frame_it->second.is_key_frame = true;
            frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose();
            frame_it->second.T = T[i];
            i++;
            continue;
        }
        // if the time of the frame is bigger than the ith time in window, add i by 1
        if((frame_it->first) > Headers[i].stamp.toSec())
        {
            i++;
        }
        // use the ith Q and T to get R_initial and P_initial(even though this frame is not the key frame)
        Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
        Vector3d P_inital = - R_inital * T[i];
        cv::eigen2cv(R_inital, tmp_r);
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(P_inital, t);

        frame_it->second.is_key_frame = false;// set is_key_frame of the ImageFrame tobe false
        vector<cv::Point3f> pts_3_vector;
        vector<cv::Point2f> pts_2_vector;        
        /*
         * map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>> > > points;
         * points is a map of a int and a vector, the int value stores the feature id, the vector stores a series of pairs, each pair store the 
         * camera id and the feature points's 3D position, 2D coordinate and velocity, here the camera id means the camera id in case of multi-camera
         */
        for (auto &id_pts : frame_it->second.points) // points is a map representing all features in the frame
        {
            int feature_id = id_pts.first;// get the feature's index
            /*
             * i_p is the element in vector<pair<int, Eigen::Matrix<double, 7, 1>> >
             * the int in pair is the camera id in case of multi-camera
             */
            for (auto &i_p : id_pts.second) 
            {
                // map<int, Vector3d> sfm_tracked_points; sfm_tracked_points is the 3d points reconstructed from the sfm part
                // sfm_tracked_points are the points constructed by GlobalSFM constrcut function
                it = sfm_tracked_points.find(feature_id);// 
                if(it != sfm_tracked_points.end())// if we can find the feature_id in the map sfm_tracked_points
                {
                    Vector3d world_pts = it->second; // get the world coordinate of the feature from the structure from motion result
                    // get 3D coordinate from the structure from motion construct result
                    cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                    pts_3_vector.push_back(pts_3);// push the world coordinate of the feature point into pts_3_vector
                    // get 2D coordinate from the frame
                    Vector2d img_pts = i_p.second.head<2>();
                    cv::Point2f pts_2(img_pts(0), img_pts(1));
                    pts_2_vector.push_back(pts_2);// push the image pixel coordinate into pts_2_vector
                }
            }
        }
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);   
        // if we can only find less than 6 feature points from the structure from motion result, print the number and print the points 
        // is not enough for solve pnp and return false  
        if(pts_3_vector.size() < 6)
        {
            cout << "pts_3_vector size " << pts_3_vector.size() << endl;
            ROS_DEBUG("Not enough points for solve pnp !");
            return false;
        }
        // if we can not solve pnp using the corresponding world coordinate and pixel coordinate,print solve pnp failed and return false
        if (! cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
        {
            ROS_DEBUG("solve pnp fail!");
            return false;
        }
        // use the pnp result to reset pose for frame_it
        cv::Rodrigues(rvec, r);
        MatrixXd R_pnp,tmp_R_pnp;
        cv::cv2eigen(r, tmp_R_pnp);
        R_pnp = tmp_R_pnp.transpose();
        MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp);
        T_pnp = R_pnp * (-T_pnp);
        frame_it->second.R = R_pnp * RIC[0].transpose();
        frame_it->second.T = T_pnp;
    }
    // if visualInitialAlign() returns true, return true, else print misalign visual structure with imu and return false
    if (visualInitialAlign())
        return true;
    else
    {
        ROS_INFO("misalign visual structure with IMU");
        return false;
    }

}

bool Estimator::visualInitialAlign()
{
    TicToc t_g;
    VectorXd x;
    //solve scale
    // align the visual and imu result, get bg, gravity and scale
    // the state vector x contains [v_b0, v_b1, ..., v_bn, g_c0, s], where s represents the scale
    bool result = VisualIMUAlignment(all_image_frame, Bgs, g, x);
    // if VisualIMUAlignment returns false, print solve g failed and return false
    if(!result)
    {
        ROS_DEBUG("solve g failed!");
        return false;
    }

    // change state
    // std_msgs::Header Headers[(WINDOW_SIZE+1)]
    // use Headers[i].stamp.toSec() to search map all_image_frame, and set the corresponding is_key_frame tobe true
    for (int i = 0; i <= frame_count; i++)
    {
        Matrix3d Ri = all_image_frame[Headers[i].stamp.toSec()].R;
        Vector3d Pi = all_image_frame[Headers[i].stamp.toSec()].T;
        Ps[i] = Pi;// Vector3d Ps[(WINDOW_SIZE+1)]
        Rs[i] = Ri;// Vector3d Rs[(WINDOW_SIZE+1)]
        all_image_frame[Headers[i].stamp.toSec()].is_key_frame = true;
    }
    // put the inverse depth of every valid feature to dep, the condition of valid is (it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2)
    VectorXd dep = f_manager.getDepthVector();
    // set every element in dep tobe -1
    for (int i = 0; i < dep.size(); i++)
        dep[i] = -1;
    f_manager.clearDepth(dep);// use dep to reset depth for every valid feature

    //triangulate on cam pose , no tic
    Vector3d TIC_TMP[NUM_OF_CAM];// NUM_OF_CAM means the multi camera case
    for(int i = 0; i < NUM_OF_CAM; i++)
        TIC_TMP[i].setZero();
    ric[0] = RIC[0];
    f_manager.setRic(ric);
    // use the equation of homogeneous coordinate x1=P1*X x2=P2*X, which is x1x(P1*X)=0,x2x(P2*X)=0, to triangulate features in the featuremanager class
    // RIC means the rotation from camera to imu
    f_manager.triangulate(Ps, &(TIC_TMP[0]), &(RIC[0]));// triangulate using the first camera in the camera chain(multi camera case)

    double s = (x.tail<1>())(0);
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        // as we do not know accelerator bias Ba now, just use 0 as Ba to integrate the imu data
        pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
    }
    for (int i = frame_count; i >= 0; i--)
        Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]);
    int kv = -1;
    map<double, ImageFrame>::iterator frame_i;
    for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++)
    {
        if(frame_i->second.is_key_frame)
        {
            kv++;
            // the state vector x contains [v_b0, v_b1, ..., v_bn, g_c0, s], where s represents the scale 
            Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
        }
    }
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        // if the feature don't satisfy the codition(used_num>=2 && start_frame<WINDOW_SIZE-2)
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        it_per_id.estimated_depth *= s;
    }

    // R0 is the rotation to transform the normalized gravity vector to (0, 0, 1)
    Matrix3d R0 = Utility::g2R(g);
    // R2ypr can get yaw, pitch, roll in degree from the 3*3 rotation matrix
    // R0*Rs[0] can rotate Rs[0] to be a coordinate whose z axis is parallel to the real world gravity
    // rot_diff = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0})*Utility::g2R(g) can rotate Rs[0] tobe a coordinate whose z axis is parallel
    // to the real time gravity and yaw angel is 0
    double yaw = Utility::R2ypr(R0 * Rs[0]).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0; // 
    g = R0 * g;
    //Matrix3d rot_diff = R0 * Rs[0].transpose();
    Matrix3d rot_diff = R0;
    for (int i = 0; i <= frame_count; i++)
    {
        Ps[i] = rot_diff * Ps[i];
        Rs[i] = rot_diff * Rs[i];
        Vs[i] = rot_diff * Vs[i];
    }
    ROS_DEBUG_STREAM("g0     " << g.transpose());
    ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose()); 

    return true;
}

/*
 * 1. for every frame in range [0, WINDOW_SIZE)
 *    1.1 go through all the feature in list feature to get all correspondings for frame i and frame WINDOW_SIZE
 *    1.2 if the number of correspondings of the two frames is bigger than 20
 *        1.2.1 compute the average disparity of all the correspondings
 *        1.2.2 if the average disparity is bigger than threshold and we succeed to solve relative r t according to the correspondings, set l to
 *              be r and return false
 * 2. if none of the frame in the window can successfully compute R and T with frame WINDOW_SIZE, return false
 */
bool Estimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l)
{
    // find previous frame which contians enough correspondance and parallex with newest frame
    // 
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        vector<pair<Vector3d, Vector3d>> corres;
        // go through all the feature in list feature to get all correspondings for frame i and frame WINDOW_SIZE
        corres = f_manager.getCorresponding(i, WINDOW_SIZE);
        // if the number of correspondings of the two frames is bigger than 20
        if (corres.size() > 20)
        {
            double sum_parallax = 0;
            double average_parallax;
            for (int j = 0; j < int(corres.size()); j++)
            {
                Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();// norm is the disparity of the corresponding pixels
                sum_parallax = sum_parallax + parallax;// add up the disparities of all the correspondings

            }
            average_parallax = 1.0 * sum_parallax / int(corres.size());// compute the average of all the disparities of the correspondings
            /*
             * if the corresponding points is more than 15 pairs:
             *    1. get E matrix using the corresponding points
             *    2. recover pose using the E matrix and corresponding points and return the inliers count, get r t from the recoverPose function
             *       if the inliers count is more than 12 return true
             */
            // if the average disparity is bigger than threshold and we succeed to solve relative r t according to the correspondings, set l to
            // be r and return true
            if(average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres, relative_R, relative_T))
            {
                l = i;
                ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
                return true;
            }
        }
    }
    // if none of frame in the window can successfully compute R and T with frame WINDOW_SIZE, return false
    return false;
}

/*
 * 1. if frame_count is less than WINDOW_SIZE, return from this function
 * 2. if solver_flag equals NON_LINEAR, conduct triangulate in the feature manager class and do the optimization
 */
void Estimator::solveOdometry()
{
    if (frame_count < WINDOW_SIZE)
        return;
    if (solver_flag == NON_LINEAR)
    {
        TicToc t_tri;
        f_manager.triangulate(Ps, tic, ric);
        ROS_DEBUG("triangulation costs %f", t_tri.toc());
        optimization();
    }
}

/*
 * 1. for every state in window, put p and q of the state to array of double para_Pose, put v, ba, bg to para_SpeedBias
 * 2. for all cameras in the camera chain, put tic and ric in queternion expression to para_Ex_Pose
 * 3. put the inverse depth of every valid feature into para_Feature
 * 4. if ESTIMATE_TD, put td into vector para_Td
 */
void Estimator::vector2double()
{
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        para_Pose[i][0] = Ps[i].x();
        para_Pose[i][1] = Ps[i].y();
        para_Pose[i][2] = Ps[i].z();
        Quaterniond q{Rs[i]};
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();

        para_SpeedBias[i][0] = Vs[i].x();
        para_SpeedBias[i][1] = Vs[i].y();
        para_SpeedBias[i][2] = Vs[i].z();

        para_SpeedBias[i][3] = Bas[i].x();
        para_SpeedBias[i][4] = Bas[i].y();
        para_SpeedBias[i][5] = Bas[i].z();

        para_SpeedBias[i][6] = Bgs[i].x();
        para_SpeedBias[i][7] = Bgs[i].y();
        para_SpeedBias[i][8] = Bgs[i].z();
    }
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        para_Ex_Pose[i][0] = tic[i].x();
        para_Ex_Pose[i][1] = tic[i].y();
        para_Ex_Pose[i][2] = tic[i].z();
        Quaterniond q{ric[i]};
        para_Ex_Pose[i][3] = q.x();
        para_Ex_Pose[i][4] = q.y();
        para_Ex_Pose[i][5] = q.z();
        para_Ex_Pose[i][6] = q.w();
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        para_Feature[i][0] = dep(i);
    if (ESTIMATE_TD)
        para_Td[0][0] = td;
}

/*
 * 
 */
void Estimator::double2vector()
{
    Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
    Vector3d origin_P0 = Ps[0];

    if (failure_occur)
    {
        origin_R0 = Utility::R2ypr(last_R0);
        origin_P0 = last_P0;
        failure_occur = 0;
    }
    // compute ypr of para_Pose
    Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                                      para_Pose[0][3],
                                                      para_Pose[0][4],
                                                      para_Pose[0][5]).toRotationMatrix());
    double y_diff = origin_R0.x() - origin_R00.x();// compute the yaw diff of originR0 and yaw of para_Pose
    //TODO
    Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
    // if one of the two angle's absolute is near to 90, print euler singular point and use matrix multiply to compute rot_diff
    if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
    {
        ROS_DEBUG("euler singular point!");
        rot_diff = Rs[0] * Quaterniond(para_Pose[0][6],
                                       para_Pose[0][3],
                                       para_Pose[0][4],
                                       para_Pose[0][5]).toRotationMatrix().transpose();
    }

    // use rot_diff to transform state in window
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {

        Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
        
        Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                para_Pose[i][1] - para_Pose[0][1],
                                para_Pose[i][2] - para_Pose[0][2]) + origin_P0;

        Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                                    para_SpeedBias[i][1],
                                    para_SpeedBias[i][2]);

        Bas[i] = Vector3d(para_SpeedBias[i][3],
                          para_SpeedBias[i][4],
                          para_SpeedBias[i][5]);

        Bgs[i] = Vector3d(para_SpeedBias[i][6],
                          para_SpeedBias[i][7],
                          para_SpeedBias[i][8]);
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d(para_Ex_Pose[i][0],
                          para_Ex_Pose[i][1],
                          para_Ex_Pose[i][2]);
        ric[i] = Quaterniond(para_Ex_Pose[i][6],
                             para_Ex_Pose[i][3],
                             para_Ex_Pose[i][4],
                             para_Ex_Pose[i][5]).toRotationMatrix();
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        dep(i) = para_Feature[i][0];
    f_manager.setDepth(dep);
    if (ESTIMATE_TD)
        td = para_Td[0][0];

    // relative info between two loop frame
    if(relocalization_info)
    { 
        Matrix3d relo_r;
        Vector3d relo_t;
        relo_r = rot_diff * Quaterniond(relo_Pose[6], relo_Pose[3], relo_Pose[4], relo_Pose[5]).normalized().toRotationMatrix();
        relo_t = rot_diff * Vector3d(relo_Pose[0] - para_Pose[0][0],
                                     relo_Pose[1] - para_Pose[0][1],
                                     relo_Pose[2] - para_Pose[0][2]) + origin_P0;
        double drift_correct_yaw;
        drift_correct_yaw = Utility::R2ypr(prev_relo_r).x() - Utility::R2ypr(relo_r).x();
        drift_correct_r = Utility::ypr2R(Vector3d(drift_correct_yaw, 0, 0));
        drift_correct_t = prev_relo_t - drift_correct_r * relo_t;   
        relo_relative_t = relo_r.transpose() * (Ps[relo_frame_local_index] - relo_t);
        relo_relative_q = relo_r.transpose() * Rs[relo_frame_local_index];
        relo_relative_yaw = Utility::normalizeAngle(Utility::R2ypr(Rs[relo_frame_local_index]).x() - Utility::R2ypr(relo_r).x());
        //cout << "vins relo " << endl;
        //cout << "vins relative_t " << relo_relative_t.transpose() << endl;
        //cout << "vins relative_yaw " <<relo_relative_yaw << endl;
        relocalization_info = 0;    

    }
}

/*
 * 1. if current frame has less than 2 feature together with previous frame, print littele feature and return true
 * 2. if the norm of last ba in wondow is bigger than 2.5, print big imu acc bias estimation and return true
 * 3. if the norm of last bg in wondow is bigger than 1, print big imu gyr bias estimation and return true
 * 4. if last pose in window substract last_P 's norm is bigger than 5, print big translation and return true
 * 5. if the z of last pose in window and z of last_P's difference is bigger than 1, print big z translation and return true
 * 6. compute the rotation between last R in window and last_R, is the delta_angle's degree value is bigger than 50, print big delta angle
 * 7. return false
 */
bool Estimator::failureDetection()
{
    if (f_manager.last_track_num < 2)// last_track_num means how many features the current frame has together with the previous frames
    {
        ROS_INFO(" little feature %d", f_manager.last_track_num);
        //return true;
    }
    if (Bas[WINDOW_SIZE].norm() > 2.5)
    {
        ROS_INFO(" big IMU acc bias estimation %f", Bas[WINDOW_SIZE].norm());
        return true;
    }
    if (Bgs[WINDOW_SIZE].norm() > 1.0)
    {
        ROS_INFO(" big IMU gyr bias estimation %f", Bgs[WINDOW_SIZE].norm());
        return true;
    }
    /*
    if (tic(0) > 1)
    {
        ROS_INFO(" big extri param estimation %d", tic(0) > 1);
        return true;
    }
    */
    Vector3d tmp_P = Ps[WINDOW_SIZE];
    if ((tmp_P - last_P).norm() > 5)
    {
        ROS_INFO(" big translation");
        return true;
    }
    if (abs(tmp_P.z() - last_P.z()) > 1)
    {
        ROS_INFO(" big z translation");
        return true; 
    }
    Matrix3d tmp_R = Rs[WINDOW_SIZE];
    Matrix3d delta_R = tmp_R.transpose() * last_R;
    Quaterniond delta_Q(delta_R);
    double delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
    if (delta_angle > 50)
    {
        ROS_INFO(" big delta_angle ");
        //return true;
    }
    return false;
}

/*
 * 
 */
void Estimator::optimization()
{
    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    //loss_function = new ceres::HuberLoss(1.0);
    loss_function = new ceres::CauchyLoss(1.0);
    // 
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        //for every state in window, para_Pose stores p and q of the state, para_SpeedBias stores v, ba, bg
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);
        problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
    }
    // for every camera in the camera chain, add its extrinsic as parameter block, if don't estimate the extrinsic, print fix extrinsic
    // and set the parameter block to constant; else print estimate extrinsic param
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);
        if (!ESTIMATE_EXTRINSIC)// if do not estimate the extrinsic, print fix extrinsic and set the parameter block to constant
        {
            ROS_DEBUG("fix extinsic param");
            problem.SetParameterBlockConstant(para_Ex_Pose[i]);
        }
        else
            ROS_DEBUG("estimate extinsic param");
    }
    // if ESTIMATE_TD, add time delay as a parameter block
    if (ESTIMATE_TD)
    {
        problem.AddParameterBlock(para_Td[0], 1);
        //problem.SetParameterBlockConstant(para_Td[0]);
    }

    TicToc t_whole, t_prepare;
    vector2double();

    // if last marginalization_info is not empty, construct new marginalization_factor, add it to optimization problem as residual block
    if (last_marginalization_info)
    {
        // construct new marginlization_factor class MarginalizationFactor : public ceres::CostFunction
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
        problem.AddResidualBlock(marginalization_factor, NULL,
                                 last_marginalization_parameter_blocks);
    }

    /* 
     * for every element in window, if the time between two elements is bigger than 10, continue to process next element,else 
     */
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        int j = i + 1;
        if (pre_integrations[j]->sum_dt > 10.0)// IntegrationBase *pre_integrations[(WINDOW_SIZE + 1)];
            continue;
        /* 
         * class IMUFactor : public ceres::SizedCostFunction<15, 7, 9, 7, 9>
         * if the size of the parameter blocks and the size of the residual vector is known at compile time(this is the common case), 
         * SizeCostFunction can be used where these values can be specified as template parameters and user only needs to implement
         * CostFunction::Evaluate()
         */
        IMUFactor* imu_factor = new IMUFactor(pre_integrations[j]);
        problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
    }
    int f_m_cnt = 0;
    int feature_index = -1;
    // for every feature in feature manager
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        // if the feature is observed by less than 2 frames or the start frame of the feature is no less than WINDOW_SIZE-2, continue to 
        // process next feature
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
 
        ++feature_index;

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        
        Vector3d pts_i = it_per_id.feature_per_frame[0].point;

        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i == imu_j)
            {
                continue;
            }
            Vector3d pts_j = it_per_frame.point;
            // if ESTIMATE_TD is true, add projection residual with td to the problem, else add projection residual without td to the problem
            if (ESTIMATE_TD)
            {
                    ProjectionTdFactor *f_td = new ProjectionTdFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                     it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td,
                                                                     it_per_id.feature_per_frame[0].uv.y(), it_per_frame.uv.y());
                    problem.AddResidualBlock(f_td, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]);
                    /*
                    double **para = new double *[5];
                    para[0] = para_Pose[imu_i];
                    para[1] = para_Pose[imu_j];
                    para[2] = para_Ex_Pose[0];
                    para[3] = para_Feature[feature_index];
                    para[4] = para_Td[0];
                    f_td->check(para);
                    */
            }
            else
            {
                ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]);
            }
            f_m_cnt++;
        }
    }

    ROS_DEBUG("visual measurement count: %d", f_m_cnt);
    ROS_DEBUG("prepare for ceres: %f", t_prepare.toc());

    if(relocalization_info)
    {
        //printf("set relocalization factor! \n");
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(relo_Pose, SIZE_POSE, local_parameterization);
        int retrive_feature_index = 0;
        int feature_index = -1;
        for (auto &it_per_id : f_manager.feature)
        {
            it_per_id.used_num = it_per_id.feature_per_frame.size();
            if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                continue;
            ++feature_index;
            int start = it_per_id.start_frame;
            if(start <= relo_frame_local_index)
            {   
                while((int)match_points[retrive_feature_index].z() < it_per_id.feature_id)
                {
                    retrive_feature_index++;
                }
                if((int)match_points[retrive_feature_index].z() == it_per_id.feature_id)
                {
                    Vector3d pts_j = Vector3d(match_points[retrive_feature_index].x(), match_points[retrive_feature_index].y(), 1.0);
                    Vector3d pts_i = it_per_id.feature_per_frame[0].point;
                    
                    ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                    problem.AddResidualBlock(f, loss_function, para_Pose[start], relo_Pose, para_Ex_Pose[0], para_Feature[feature_index]);
                    retrive_feature_index++;
                }     
            }
        }

    }

    ceres::Solver::Options options;

    options.linear_solver_type = ceres::DENSE_SCHUR;
    //options.num_threads = 2;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = NUM_ITERATIONS;
    //options.use_explicit_schur_complement = true;
    //options.minimizer_progress_to_stdout = true;
    //options.use_nonmonotonic_steps = true;
    if (marginalization_flag == MARGIN_OLD)
        options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;
    else
        options.max_solver_time_in_seconds = SOLVER_TIME;
    TicToc t_solver;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    //cout << summary.BriefReport() << endl;
    ROS_DEBUG("Iterations : %d", static_cast<int>(summary.iterations.size()));
    ROS_DEBUG("solver costs: %f", t_solver.toc());

    double2vector();

    TicToc t_whole_marginalization;
    if (marginalization_flag == MARGIN_OLD)
    {
        MarginalizationInfo *marginalization_info = new MarginalizationInfo();
        vector2double();

        if (last_marginalization_info)
        {
            vector<int> drop_set;
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
            {
                if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
                    last_marginalization_parameter_blocks[i] == para_SpeedBias[0])
                    drop_set.push_back(i);
            }
            // construct new marginlization_factor
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                           last_marginalization_parameter_blocks,
                                                                           drop_set);

            marginalization_info->addResidualBlockInfo(residual_block_info);
        }

        {
            if (pre_integrations[1]->sum_dt < 10.0)
            {
                IMUFactor* imu_factor = new IMUFactor(pre_integrations[1]);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factor, NULL,
                                                                           vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]},
                                                                           vector<int>{0, 1});
                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
        }

        {
            int feature_index = -1;
            for (auto &it_per_id : f_manager.feature)
            {
                it_per_id.used_num = it_per_id.feature_per_frame.size();
                if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                    continue;

                ++feature_index;

                int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
                if (imu_i != 0)
                    continue;

                Vector3d pts_i = it_per_id.feature_per_frame[0].point;

                for (auto &it_per_frame : it_per_id.feature_per_frame)
                {
                    imu_j++;
                    if (imu_i == imu_j)
                        continue;

                    Vector3d pts_j = it_per_frame.point;
                    if (ESTIMATE_TD)
                    {
                        ProjectionTdFactor *f_td = new ProjectionTdFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                          it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td,
                                                                          it_per_id.feature_per_frame[0].uv.y(), it_per_frame.uv.y());
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f_td, loss_function,
                                                                                        vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]},
                                                                                        vector<int>{0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                    else
                    {
                        ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                       vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]},
                                                                                       vector<int>{0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                }
            }
        }

        TicToc t_pre_margin;
        marginalization_info->preMarginalize();
        ROS_DEBUG("pre marginalization %f ms", t_pre_margin.toc());
        
        TicToc t_margin;
        marginalization_info->marginalize();
        ROS_DEBUG("marginalization %f ms", t_margin.toc());

        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i <= WINDOW_SIZE; i++)
        {
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
            addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
        }
        for (int i = 0; i < NUM_OF_CAM; i++)
            addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
        if (ESTIMATE_TD)
        {
            addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
        }
        vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

        if (last_marginalization_info)
            delete last_marginalization_info;
        last_marginalization_info = marginalization_info;
        last_marginalization_parameter_blocks = parameter_blocks;
        
    }
    else
    {
        if (last_marginalization_info &&
            std::count(std::begin(last_marginalization_parameter_blocks), std::end(last_marginalization_parameter_blocks), para_Pose[WINDOW_SIZE - 1]))
        {

            MarginalizationInfo *marginalization_info = new MarginalizationInfo();
            vector2double();
            if (last_marginalization_info)
            {
                vector<int> drop_set;
                for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
                {
                    ROS_ASSERT(last_marginalization_parameter_blocks[i] != para_SpeedBias[WINDOW_SIZE - 1]);
                    if (last_marginalization_parameter_blocks[i] == para_Pose[WINDOW_SIZE - 1])
                        drop_set.push_back(i);
                }
                // construct new marginlization_factor
                MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                               last_marginalization_parameter_blocks,
                                                                               drop_set);

                marginalization_info->addResidualBlockInfo(residual_block_info);
            }

            TicToc t_pre_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->preMarginalize();
            ROS_DEBUG("end pre marginalization, %f ms", t_pre_margin.toc());

            TicToc t_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->marginalize();
            ROS_DEBUG("end marginalization, %f ms", t_margin.toc());
            
            // unordered_map containers are faster than map containers to access individual elements by their key, although they are
            // generally less efficient for range iteration through a subset of their elements
            std::unordered_map<long, double *> addr_shift;
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                if (i == WINDOW_SIZE - 1)
                    continue;
                else if (i == WINDOW_SIZE)
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
                }
                else
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
                }
            }
            for (int i = 0; i < NUM_OF_CAM; i++)
                addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
            if (ESTIMATE_TD)
            {
                addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
            }
            
            vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
            if (last_marginalization_info)
                delete last_marginalization_info;
            last_marginalization_info = marginalization_info;
            last_marginalization_parameter_blocks = parameter_blocks;
            
        }
    }
    ROS_DEBUG("whole marginalization costs: %f", t_whole_marginalization.toc());
    
    ROS_DEBUG("whole time for ceres: %f", t_whole.toc());
}

void Estimator::slideWindow()
{
    TicToc t_margin;
    if (marginalization_flag == MARGIN_OLD)
    {
        back_R0 = Rs[0];
        back_P0 = Ps[0];
        if (frame_count == WINDOW_SIZE)
        {
            for (int i = 0; i < WINDOW_SIZE; i++)
            {
                Rs[i].swap(Rs[i + 1]);

                std::swap(pre_integrations[i], pre_integrations[i + 1]);

                dt_buf[i].swap(dt_buf[i + 1]);
                linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
                angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

                Headers[i] = Headers[i + 1];
                Ps[i].swap(Ps[i + 1]);
                Vs[i].swap(Vs[i + 1]);
                Bas[i].swap(Bas[i + 1]);
                Bgs[i].swap(Bgs[i + 1]);
            }
            Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
            Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
            Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
            Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];
            Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
            Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

            delete pre_integrations[WINDOW_SIZE];
            pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();

            if (true || solver_flag == INITIAL)
            {
                double t_0 = Headers[0].stamp.toSec();
                map<double, ImageFrame>::iterator it_0;
                it_0 = all_image_frame.find(t_0);
                delete it_0->second.pre_integration;
                all_image_frame.erase(all_image_frame.begin(), it_0);

            }
            slideWindowOld();
        }
    }
    else
    {
        if (frame_count == WINDOW_SIZE)
        {
            for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++)
            {
                double tmp_dt = dt_buf[frame_count][i];
                Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
                Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count][i];

                pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

                dt_buf[frame_count - 1].push_back(tmp_dt);
                linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
                angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
            }

            Headers[frame_count - 1] = Headers[frame_count];
            Ps[frame_count - 1] = Ps[frame_count];
            Vs[frame_count - 1] = Vs[frame_count];
            Rs[frame_count - 1] = Rs[frame_count];
            Bas[frame_count - 1] = Bas[frame_count];
            Bgs[frame_count - 1] = Bgs[frame_count];

            delete pre_integrations[WINDOW_SIZE];
            pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();

            slideWindowNew();
        }
    }
}

// real marginalization is removed in solve_ceres()
void Estimator::slideWindowNew()
{
    sum_of_front++;
    f_manager.removeFront(frame_count);
}
// real marginalization is removed in solve_ceres()
void Estimator::slideWindowOld()
{
    sum_of_back++;

    bool shift_depth = solver_flag == NON_LINEAR ? true : false;
    if (shift_depth)
    {
        Matrix3d R0, R1;
        Vector3d P0, P1;
        R0 = back_R0 * ric[0];
        R1 = Rs[0] * ric[0];
        P0 = back_P0 + back_R0 * tic[0];
        P1 = Ps[0] + Rs[0] * tic[0];
        f_manager.removeBackShiftDepth(R0, P0, R1, P1);
    }
    else
        f_manager.removeBack();
}

void Estimator::setReloFrame(double _frame_stamp, int _frame_index, vector<Vector3d> &_match_points, Vector3d _relo_t, Matrix3d _relo_r)
{
    // set time, index and matched points for the reloc frame
    // give the pose value in the function parameter list to prev_relo_t and prev_relo_r 
    relo_frame_stamp = _frame_stamp;
    relo_frame_index = _frame_index;
    match_points.clear();
    match_points = _match_points;
    prev_relo_t = _relo_t;
    prev_relo_r = _relo_r;
    for(int i = 0; i < WINDOW_SIZE; i++)// for all poses in the window
    {
        if(relo_frame_stamp == Headers[i].stamp.toSec())// if one stamp is equal to the relo_frame_stamp
        {
            relo_frame_local_index = i;//set the reloc_frame_local_index tobe i
            relocalization_info = 1;// set the bool relocalization_info tobe true
            for (int j = 0; j < SIZE_POSE; j++)//set the reloc pose tobe the pose in para_Pose
                relo_Pose[j] = para_Pose[i][j];
        }
    }
}

