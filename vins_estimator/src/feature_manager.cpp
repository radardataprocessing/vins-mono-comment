#include "feature_manager.h"

int FeaturePerId::endFrame()// I think this means the index of the lastest frame observing the feature
{
    /* every element in feature_per_frame contains a feature observed by a frame, I think vector<FeaturePerFrame> feature_per_frame stores 
       the information of a feature observed by different frames*/
    return start_frame + feature_per_frame.size() - 1;
}

FeatureManager::FeatureManager(Matrix3d _Rs[])
    : Rs(_Rs) // Rs mean the rotation from imu to world
{
    for (int i = 0; i < NUM_OF_CAM; i++) // NUM_OF_CAM represents how many cameras are there on the hardware platform
        ric[i].setIdentity();
}

// ric is an array of Matrix3d, the size of the array is NUM_OF_CAM
void FeatureManager::setRic(Matrix3d _ric[])
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ric[i] = _ric[i];
    }
}

void FeatureManager::clearState()// clear the list feature
{
    feature.clear(); // list<FeaturePerId> feature 
}

/*
 * get the number of valid features
 */
int FeatureManager::getFeatureCount()
{
    int cnt = 0;
    for (auto &it : feature)
    {

        it.used_num = it.feature_per_frame.size();

        if (it.used_num >= 2 && it.start_frame < WINDOW_SIZE - 2)
        {
            cnt++;
        }
    }
    return cnt;
}

/*
 * image is a map of a int and a vector, the int value stores the feature id, the vector stores a series of pairs, each pair store the camera id and 
 * the feature point's 3D position, 2D coordinate and velocity, here the camera id means the camera id in case of multi-camera 
 * I think the map image in the parameter list means all features in one frame, frame_count is the index of the image
 * 
 * 1. for every feature point in image, use the feature_id to find whether the feature is already in the list<FeaturePerId> feature, if
 *    no, push feature(id is input feature_id, start_frame is feature_count) to list feature, and push the FeaturePerFrame object
 *    to the constructed FeaturePerId object; else just push the FeaturePerFrame object to the found index and add 1 to last_track_num
 * 2. if frame count is less than 2 or last_track_num is less than 20, return true
 * 3. for every FeaturePerId object in vector feature, if the start_frame is no bigger than frame_count-2 and the endFrame is no less 
 *    than frame_count -1, add disparity of the feature to parallax_sum, add 1 to parallax_num
 * 4. if parallax_num equals 0, return true; else, compute average parallax as parallax_sum/parallax_num, return average>=MIN_PARALLAX
 */
bool FeatureManager::addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td)
{
    ROS_DEBUG("input feature: %d", (int)image.size());
    ROS_DEBUG("num of feature: %d", getFeatureCount());
    double parallax_sum = 0;
    int parallax_num = 0;
    last_track_num = 0;// last_track_num means how many features the current frame has together with the previous frames
    for (auto &id_pts : image)// for all feature points in a frame
    {
        // get the x,y,z,u,v,velocity of the feature observe by the first camera and td to construct a FeaturePerFrame object
        FeaturePerFrame f_per_fra(id_pts.second[0].second, td);

        int feature_id = id_pts.first;// get the id of the feature
        // list<FeaturePerId> feature   find whether here's an element in list feature who has the same feature id with the current feature point
        // find whether here is a feature who has the same id with current feature in the list feature
        /* 
         * template <class InputIterator, class UnaryPredicate>
         * InputIterator find_if (InputIterator first, InputIterator last, UnaryPredicate pred)
         * returns an iterator to the first element in the range [first, last) for which pred returns true. If no such element is 
         * found, the function returns last
         */
        auto it = find_if(feature.begin(), feature.end(), [feature_id](const FeaturePerId &it)
                          {
            return it.feature_id == feature_id;
                          });
        // if we can't find the element in the list feature, construct one and push it into the list, and push f_per_fra into the newly constructed feature
        if (it == feature.end())// if we can not find the element in list feature, construct one and push it into the list
        {
            // the second parameter in the constructor of FeaturePerId is used as the start_frame of the feature
            feature.push_back(FeaturePerId(feature_id, frame_count)); // FeaturePerId(int _feature_id, int _start_frame)
            feature.back().feature_per_frame.push_back(f_per_fra);
        }
        // if we can find feature having the same id as the current feature, push f_per_fra to the feature we found in list feature and add last_track_num by 1 
        else if (it->feature_id == feature_id)// if we can find the element, push f_per_fra to the element vector and add last_track_num by 1
        {
            it->feature_per_frame.push_back(f_per_fra);
            last_track_num++;
        }
    }

    // if the frame_count is less than 2 or the last_track_num is less than 20, return true
    /*
     * frame count is the frame id of the given frame, if frame_count is less than 2 or the last_track_num is less than 20, return true
     */
    if (frame_count < 2 || last_track_num < 20)
        return true;

    for (auto &it_per_id : feature)// for every element in the list feature
    {
        // if the start_frame is no bigger than frame_count-2 and the endFrame is no less than frame_count -1
        if (it_per_id.start_frame <= frame_count - 2 &&
            it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) - 1 >= frame_count - 1)
        {
            // get the parallax between the second last frame(whose index is frame_count - 1 -it_per_id.start_frame) and third last 
            // frame(whose index is frame_count - 2 -it_per_id.start_frame)
            parallax_sum += compensatedParallax2(it_per_id, frame_count);// parallax means disparity
            parallax_num++;
        }
    }
    // if no feature in the feature list has start_frame no bigger than frame_count-2 and endFrame no less than frame_count -1, return true
    if (parallax_num == 0)
    {
        return true;
    }
    else
    {
        ROS_DEBUG("parallax_sum: %lf, parallax_num: %d", parallax_sum, parallax_num);
        ROS_DEBUG("current parallax: %lf", parallax_sum / parallax_num * FOCAL_LENGTH);
        return parallax_sum / parallax_num >= MIN_PARALLAX;// if the average parallax is no less than MIN_PARALLAX, return true
    }
}

void FeatureManager::debugShow()
{
    ROS_DEBUG("debug show");
    for (auto &it : feature)
    {
        ROS_ASSERT(it.feature_per_frame.size() != 0);// check whether here's actually at least one frame observing the feature
        ROS_ASSERT(it.start_frame >= 0);// check whether the start_frame of the feature is no less than 0
        ROS_ASSERT(it.used_num >= 0);// check whether the used_num of the feature is no less than 0

        ROS_DEBUG("%d,%d,%d ", it.feature_id, it.used_num, it.start_frame);// print the feature_id, used_num and start_frame of the feature
        int sum = 0;
        for (auto &j : it.feature_per_frame)
        {
            ROS_DEBUG("%d,", int(j.is_used));
            sum += j.is_used;
            printf("(%lf,%lf) ",j.point(0), j.point(1));
        }
        ROS_ASSERT(it.used_num == sum);
    }
}

/*
 * I think here the frame_count_l and frame_count_r represents two frames we are interested, if they are in the range of feature_per_frame, get 3D 
 * coordinate for the two frames, go through all the feature in list feature to get all correspondings for the two frames
 * 
 * for every FeaturePerId object in list feature, if [frame_count_l, frame_count r] is a subset of frames observing the feature, use their frame_count
 * to compute their index in the vector feature_per_frame, get the corresponding 3d coordinates, push the pair into vector corres
 */
vector<pair<Vector3d, Vector3d>> FeatureManager::getCorresponding(int frame_count_l, int frame_count_r)
{
    vector<pair<Vector3d, Vector3d>> corres;
    for (auto &it : feature)
    {
        // makesure that this feature can be observed by frame_count_l and frame_count_r
        // which means [frame_count_l, frame_count r] is a subset of frames observing the feature
        if (it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r)// endFrame=start_frame + feature_per_frame.size()-1
        {
            Vector3d a = Vector3d::Zero(), b = Vector3d::Zero();
            int idx_l = frame_count_l - it.start_frame;
            int idx_r = frame_count_r - it.start_frame;

            a = it.feature_per_frame[idx_l].point;

            b = it.feature_per_frame[idx_r].point;
            
            corres.push_back(make_pair(a, b));
        }
    }
    return corres;
}

/*
 * for every valid feature in the list feature, set its estimated_depth using the parameter of the function, if the estimated_depth of the feature is negative,
 * set solve_flag of the feature to 2; else set the solve_flag of the feature to 1
 * 
 */
void FeatureManager::setDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        it_per_id.estimated_depth = 1.0 / x(++feature_index);
        //ROS_INFO("feature id %d , start_frame %d, depth %f ", it_per_id->feature_id, it_per_id-> start_frame, it_per_id->estimated_depth);
        if (it_per_id.estimated_depth < 0)
        {
            it_per_id.solve_flag = 2;
        }
        else
            it_per_id.solve_flag = 1;
    }
}

/*
 * for every feature in list feature, if the solve_flag of the feature equals 2, erase it from the list feature 
 */
void FeatureManager::removeFailures()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        if (it->solve_flag == 2)// if the depth of a feature is less than 0, set the solve flag to 2, else the solve flag of a feature is 1
            feature.erase(it);
    }
}

/*
 * set depth for every valid feature in list feature
 * the difference between this function and setDepth is that this function do not set solve_flag for the feature
 */
void FeatureManager::clearDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        it_per_id.estimated_depth = 1.0 / x(++feature_index);
    }
}

/*
 * put the inverse depth of every valid feature, the condition of valid is (it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2) to the 
 * returned VectorXd
 * get the inverse depth of valid features in list feature
 */
VectorXd FeatureManager::getDepthVector()
{
    // I think getFeatureCount() returns the valid feature num in the list feature
    VectorXd dep_vec(getFeatureCount());
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        // this condition in the if block is the same as that in the getFeatureCount function
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
#if 1
        dep_vec(++feature_index) = 1. / it_per_id.estimated_depth;// put the inverse depth of the feature to the VectorXd
#else
        dep_vec(++feature_index) = it_per_id->estimated_depth;
#endif
    }
    return dep_vec;
}

/*
 * denote the homogeneous coordinate of the same 3D point X in image coordinate as x1 and x2, and their corresponding projection matrix as P1,P2, we have
 * x1 = P1*X   x2=P2*X
 * writing in the cross-product form is            x1x(P1*X) = 0           x2x(P2*X) = 0
 * then, we write it into matrix form as
 * |  0 -z  y |   | P1T*X |
 * |  z  0 -x | * | P2T*X | = 0
 * | -y  x  0 |   | P3T*X |
 * the first row of the matrix can be writen as       -z*P2T*X + y*P3T*X = 0   (1)
 * the second row of the matrix can be writen as      z*P1T*X - x*P3T*X = 0    (2)
 * the third row of the matrix can be writen as       -y*P1T*X + x*P2T*X = 0   (3)
 * (1)*x + (2)*y, we can get -z*x*P2T*X+z*y*P1T*X = 0 which is -z times equastion(3)
 * so we can get 2 equations with one point projection, with a pair of projection, we can get 4 equations, the number of equation is more than the number of 
 * variables, so we can not directly solve the problem. We can apply SVD decompotion for the parameter matrix, finally we can get the vector corresponding to the 
 * minimum singular , if X=(x,y,z,w), the depth is z/w
 * 
 * 
 * for every FeaturePerId object it_per_id in list feature
 *    1. assign used_num of it_per_id using it_per_id.feature_per_frame.size()
 *    2. if this is not a valid feature, continue to process next feature
 *    3. if the feature already have an estimated_depth, continue to process next feature
 *    4. use frames between the start frame of the feature and every other frame to get the traingulate equation above, the svd matrix has
 *       (2 * it_per_id.feature_per_frame.size()) rows and 4 cols
 *    5. conduct svd decomposition for the matrix above and get the 4 dimension solution svd_V
 *    6. get depth as svd_V[2]/svd_V[3], if the depth is smaller than 0.1, set the depth of this feature tobe INIT_DEPTH
 */
void FeatureManager::triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[])
{
    for (auto &it_per_id : feature)// for every feature in the feature list
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();// the used_num of the feature is the size of feature_per_frame
        /*
         * if the feature don't satisfy the following two conditions at the same time:
         * 1. the used_num of the feature is bigger than 2
         * 2. the start_frame is less than WINDOW_SIZE-2
         * continue to process next feature
         */
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        if (it_per_id.estimated_depth > 0)// if the feature already have an estimated_depth, continue to process next feature
            continue;
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

        ROS_ASSERT(NUM_OF_CAM == 1);
        //every feature has 2 equtions, each equation has 4 parameters to be solved, so the matrix has 2*it_per_id.feature_per_frame.size() rows and 4 cols
        Eigen::MatrixXd svd_A(2 * it_per_id.feature_per_frame.size(), 4);
        int svd_idx = 0;

        Eigen::Matrix<double, 3, 4> P0;
        Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];
        Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];
        P0.leftCols<3>() = Eigen::Matrix3d::Identity();
        P0.rightCols<1>() = Eigen::Vector3d::Zero();

        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            // tic is an array of Vector3d, ric is an array of Matrix3d
            /*
             * here, Ps and Rs means the imu to world position and imu to world roration
             * R_cam_to_world = R_imu_to_world * R_cam_to_imu
             * t_cam_to_world = t_imu_to_world + t_cam_to_imu_under_world_coordinate
             *                = t_imu_to_world + R_imu_to_world*t_cam_to_imu
             * R_cam2_to_cam1 = R_world_to_cam1 * R_cam2_to_world
             *                = R_cam1_to_world.inverse() * R_cam2_to_world
             * t_cam2_to_cam1 = R_world_to_cam1 * t_cam2_to_cam1_under_world_coordinate
             *                = R_world_to_cam1 * (t_cam2_to_world - t_cam1_to_world)
             *                = R_cam1_to_world * (t_cam2_to_world - t_cam1_to_world)
             */
            Eigen::Vector3d t1 = Ps[imu_j] + Rs[imu_j] * tic[0];
            Eigen::Matrix3d R1 = Rs[imu_j] * ric[0];
            Eigen::Vector3d t = R0.transpose() * (t1 - t0);
            Eigen::Matrix3d R = R0.transpose() * R1;
            Eigen::Matrix<double, 3, 4> P;
            P.leftCols<3>() = R.transpose();
            P.rightCols<1>() = -R.transpose() * t;
            Eigen::Vector3d f = it_per_frame.point.normalized();// f is the 3D coordinate in the frame coordinate
            svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
            svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);

            if (imu_i == imu_j)
                continue;
        }
        ROS_ASSERT(svd_idx == svd_A.rows());
        // get the singular vector corresponding to the minimal singular value as the solve of svd_A*x=0
        Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
        // the homogeneous coordinate solved by svd is (x,y,z,w), the depth of the point is z/w
        double svd_method = svd_V[2] / svd_V[3];
        //it_per_id->estimated_depth = -b / A;
        //it_per_id->estimated_depth = svd_V[2] / svd_V[3];

        it_per_id.estimated_depth = svd_method;
        //it_per_id->estimated_depth = INIT_DEPTH;

        // if the depth solved by the SVD decomposion is less than 0.1, set the estimated_depth tobe the INIT_DEPTH
        if (it_per_id.estimated_depth < 0.1)
        {
            it_per_id.estimated_depth = INIT_DEPTH;
        }

    }
}

// for every feature in list feature, if the used_num is not 0 and the feature is marked as an outlier, erase the feature from list
void FeatureManager::removeOutlier()
{
    ROS_BREAK();
    int i = -1;
    for (auto it = feature.begin(), it_next = feature.begin(); // for every feature in the list feature
         it != feature.end(); it = it_next)
    {
        it_next++;
        i += it->used_num != 0;
        if (it->used_num != 0 && it->is_outlier == true)
        {
            feature.erase(it);
        }
    }
}

// I think this function may be invoked when we need to marginalize the old frame in the window
/**
 * for every FeaturePerId object in list feature
 *    1. if the start_frame of the frame is not 0, substract 1 from start_frame
 *    2. else(start_frame of the iterator equals 0)
 *       2.1 get the point coordinate from feature_per_frame[0]
 *       2.2 remove feature_per_frame[0] from feature_per_frame
 *       2.3 if feature_per_frame.size() < 2, erase this feature from list feature
 *       2.4 transform the point coordinate from marg coordinate to new coordinate, get z value of the transformd point,
 *           if the z is bigger than 0, assign estimated_depth using the transformd z value; else assign estimated_depth using the INIT_DEPTH
 */
void FeatureManager::removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P)
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)// for every feature in the list feature
    {
        it_next++;// maybe it will be removed from list feature, so just backup the next iterator in advance

        if (it->start_frame != 0)// if the start_frame is not the first frame
            it->start_frame--;
        else// if the function goes into this branch, it means the start frame of this feature is 0, which also means the start frame is the first frame in the window
        {
            Eigen::Vector3d uv_i = it->feature_per_frame[0].point;  
            it->feature_per_frame.erase(it->feature_per_frame.begin());// erase the feature observed by the first frame in vector feature_per_frame
            // after erase the feature observed by the first frame, if the vector feature_per_frame has more than 2 elements, then just erase this feature from the 
            // list feature  
            if (it->feature_per_frame.size() < 2)
            {
                feature.erase(it);
                continue;
            }
            /*
             * Pw = R_cam_to_world * Pc + t_cam_to_world
             * R_cam_to_wolrd.inverse * Pw = Pc + R_cam_to_wolrd.inverse * t_cam_to_world
             * Pc = R_cam_to_world.inverse * Pw - R_cam_to_wolrd.inverse * t_cam_to_world
             * from the above equations 
             * R_world_to_cam = R_cam_to_world.inverse = R_cam_to_world.transpose
             * t_world_to_cam = - R_cam_to_wolrd.inverse * t_cam_to_world = - R_cam_to_wolrd.transpose * t_cam_to_world
             */
            else
            {
                Eigen::Vector3d pts_i = uv_i * it->estimated_depth;// get the point in the camera coordinate
                Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P; // transpose the point from the marg coordinate to world coordinate
                Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P);// transpose the point from the world coordinate to camera coordinate
                double dep_j = pts_j(2);// get the depth in the new coordinate
                if (dep_j > 0)// if the depth in the new coordinate is positive, set the estimated_depth of this feature tobe the depth in the new frame
                    it->estimated_depth = dep_j;
                else// if the depth in the new coordinate is not positive, set the estimated_depth of this feature tobe INIT_DEPTH
                    it->estimated_depth = INIT_DEPTH;
            }
        }
        // remove tracking-lost feature after marginalize
        /*
        if (it->endFrame() < WINDOW_SIZE - 1)
        {
            feature.erase(it);
        }
        */
    }
}

/**
 * for every FramePerId object in list feature
 *    1. if start_frame do not equal 0, substract 1 from start_frame
 *    2. else(start_frame equal 0), erase first element from feature_per_frame, if it->feature_per_frame.size() equals 
 *       0, remove this feature from list feature
 */
void FeatureManager::removeBack()// I think here back means the oldest feature
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            if (it->feature_per_frame.size() == 0)// if the feature with the certain id has no frame observing it, erase the element from list feature
                feature.erase(it);
        }
    }
}

void FeatureManager::removeFront(int frame_count)// I think front here means the newest feature
{
    // for every feature in list feature
    for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next)// list<FeaturePerId> feature
    {
        it_next++;

        if (it->start_frame == frame_count)// if the start frame of the feature equals frame count, substract the start_frame of the feature by 1
        {
            it->start_frame--;
        }
        else
        {
            // I think here j means the index of the last frame in the sliding window in the vector feature_per_frame
            int j = WINDOW_SIZE - 1 - it->start_frame;
            //it->endFrame()=it->start_frame + feature_per_frame.size()-1
            if (it->endFrame() < frame_count - 1)
                continue;
            it->feature_per_frame.erase(it->feature_per_frame.begin() + j);
            if (it->feature_per_frame.size() == 0)// if the feature with the certain id has no frame observing it, erase the element from list feature
                feature.erase(it);
        }
    }
}

/*
 * the variable it_per_id in the parameter list stores features with the same id extracted from different frames, get the feature in the second last frame 
 * and third last frame, then compute the parallax between them
 */
double FeatureManager::compensatedParallax2(const FeaturePerId &it_per_id, int frame_count)
{
    //check the second last frame is keyframe or not
    //parallax betwwen seconde last frame and third last frame
    const FeaturePerFrame &frame_i = it_per_id.feature_per_frame[frame_count - 2 - it_per_id.start_frame];
    const FeaturePerFrame &frame_j = it_per_id.feature_per_frame[frame_count - 1 - it_per_id.start_frame];

    double ans = 0;
    Vector3d p_j = frame_j.point;

    double u_j = p_j(0);
    double v_j = p_j(1);

    Vector3d p_i = frame_i.point;
    Vector3d p_i_comp;

    //int r_i = frame_count - 2;
    //int r_j = frame_count - 1;
    //p_i_comp = ric[camera_id_j].transpose() * Rs[r_j].transpose() * Rs[r_i] * ric[camera_id_i] * p_i;
    p_i_comp = p_i;
    double dep_i = p_i(2);
    double u_i = p_i(0) / dep_i;
    double v_i = p_i(1) / dep_i;
    double du = u_i - u_j, dv = v_i - v_j;

    double dep_i_comp = p_i_comp(2);
    double u_i_comp = p_i_comp(0) / dep_i_comp;
    double v_i_comp = p_i_comp(1) / dep_i_comp;
    double du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j;

    ans = max(ans, sqrt(min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp)));

    return ans;
}