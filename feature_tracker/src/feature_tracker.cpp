#include "feature_tracker.h"

// frames as the time goes by:  prev->cur->forw

int FeatureTracker::n_id = 0;

// check whether the point is in the rect defined by ROW, COL, BORDER_SIZE
bool inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
}

/*
 * every element in v has a corresponding stat in status, if status is ok, put the element in the new vector, else delete the element
 */
void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}


FeatureTracker::FeatureTracker()
{
}

void FeatureTracker::setMask()
{
    if(FISHEYE)
        mask = fisheye_mask.clone();
    else
        mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));
    

    // prefer to keep features that are tracked for long time
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

    // the vector cnt_pts_id stores the information  pair<track_count, pair<point, point_id>>
    for (unsigned int i = 0; i < forw_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(forw_pts[i], ids[i])));

    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
         {
            return a.first > b.first;
         });// sort the cnt_pts_id according to the track count

    forw_pts.clear();
    ids.clear();
    track_cnt.clear();

    for (auto &it : cnt_pts_id)// for all points in cnt_pts_id
    {
        if (mask.at<uchar>(it.second.first) == 255)// if the mask on the feature point is 255 
        {
            forw_pts.push_back(it.second.first);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
        }
    }
}

void FeatureTracker::addPoints()// add point into vector forw_pts, ids and track_cnt, the id was set to -1 and the track count was set to 1
{
    for (auto &p : n_pts)
    {
        forw_pts.push_back(p);
        ids.push_back(-1);
        track_cnt.push_back(1);
    }
}

void FeatureTracker::readImage(const cv::Mat &_img, double _cur_time)
{
    cv::Mat img;
    TicToc t_r;
    cur_time = _cur_time;

    if (EQUALIZE)
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        TicToc t_c;
        clahe->apply(_img, img);
        ROS_DEBUG("CLAHE costs: %fms", t_c.toc());
    }
    else
        img = _img;

    if (forw_img.empty())
    {
        prev_img = cur_img = forw_img = img;
    }
    else
    {
        forw_img = img;
    }

    forw_pts.clear();

    if (cur_pts.size() > 0)
    {
        TicToc t_o;
        vector<uchar> status;
        vector<float> err;
        /** @brief Calculates an optical flow for a sparse feature set using the iterative Lucas-Kanade method with
        pyramids.

        @param prevImg first 8-bit input image or pyramid constructed by buildOpticalFlowPyramid.
        @param nextImg second input image or pyramid of the same size and the same type as prevImg.
        @param prevPts vector of 2D points for which the flow needs to be found; point coordinates must be
        single-precision floating-point numbers.
        @param nextPts output vector of 2D points (with single-precision floating-point coordinates)
        containing the calculated new positions of input features in the second image; when
        OPTFLOW_USE_INITIAL_FLOW flag is passed, the vector must have the same size as in the input.
        @param status output status vector (of unsigned chars); each element of the vector is set to 1 if
        the flow for the corresponding features has been found, otherwise, it is set to 0.
        @param err output vector of errors; each element of the vector is set to an error for the
        corresponding feature, type of the error measure can be set in flags parameter; if the flow wasn't
        found then the error is not defined (use the status parameter to find such cases).
        @param winSize size of the search window at each pyramid level.
        @param maxLevel 0-based maximal pyramid level number; if set to 0, pyramids are not used (single
        level), if set to 1, two levels are used, and so on; if pyramids are passed to input then
        algorithm will use as many levels as pyramids have but no more than maxLevel.
        @param criteria parameter, specifying the termination criteria of the iterative search algorithm
        (after the specified maximum number of iterations criteria.maxCount or when the search window
        moves by less than criteria.epsilon.
        @param flags operation flags:
         -   **OPTFLOW_USE_INITIAL_FLOW** uses initial estimations, stored in nextPts; if the flag is
             not set, then prevPts is copied to nextPts and is considered the initial estimate.
         -   **OPTFLOW_LK_GET_MIN_EIGENVALS** use minimum eigen values as an error measure (see
             minEigThreshold description); if the flag is not set, then L1 distance between patches
             around the original and a moved point, divided by number of pixels in a window, is used as a
             error measure.
        @param minEigThreshold the algorithm calculates the minimum eigen value of a 2x2 normal matrix of
        optical flow equations (this matrix is called a spatial gradient matrix in @cite Bouguet00), divided
        by number of pixels in a window; if this value is less than minEigThreshold, then a corresponding
        feature is filtered out and its flow is not processed, so it allows to remove bad points and get a
        performance boost.

        The function implements a sparse iterative version of the Lucas-Kanade optical flow in pyramids. See
        @cite Bouguet00 . The function is parallelized with the TBB library.

        @note

        -   An example using the Lucas-Kanade optical flow algorithm can be found at
            opencv_source_code/samples/cpp/lkdemo.cpp
        -   (Python) An example using the Lucas-Kanade optical flow algorithm can be found at
            opencv_source_code/samples/python/lk_track.py
        -   (Python) An example using the Lucas-Kanade tracker for homography matching can be found at
            opencv_source_code/samples/python/lk_homography.py
         
        CV_EXPORTS_W void calcOpticalFlowPyrLK( InputArray prevImg, InputArray nextImg,
                                                InputArray prevPts, InputOutputArray nextPts,
                                                OutputArray status, OutputArray err,
                                                Size winSize = Size(21,21), int maxLevel = 3,
                                                TermCriteria criteria = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01),
                                                int flags = 0, double minEigThreshold = 1e-4 );*/
        cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3);

        for (int i = 0; i < int(forw_pts.size()); i++)// for every point in forw_pts, if the status is ok but the point is not in the border, set status to bad 
            if (status[i] && !inBorder(forw_pts[i]))
                status[i] = 0;
        // remove elements acoording to the optical flow status
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(ids, status);
        reduceVector(cur_un_pts, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
    }

    for (auto &n : track_cnt)// add the track count for every point by 1
        n++;

    if (PUB_THIS_FRAME)
    {
        // compute fundamental matrix using a set of corresponding points, use the status of fundamental computation to delete some elements in the vector
        rejectWithF();
        ROS_DEBUG("set mask begins");
        TicToc t_m;
        setMask();// select those points that are in the valid area defined by the mask
        ROS_DEBUG("set mask costs %fms", t_m.toc());

        ROS_DEBUG("detect feature begins");
        TicToc t_t;
        int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());
        if (n_max_cnt > 0)
        {
            if(mask.empty())
                cout << "mask is empty " << endl;
            if (mask.type() != CV_8UC1)
                cout << "mask type wrong " << endl;
            if (mask.size() != forw_img.size())
                cout << "wrong size " << endl;
            cv::goodFeaturesToTrack(forw_img, n_pts, MAX_CNT - forw_pts.size(), 0.01, MIN_DIST, mask);
        }
        else
            n_pts.clear();
        ROS_DEBUG("detect feature costs: %fms", t_t.toc());

        ROS_DEBUG("add feature begins");
        TicToc t_a;
        addPoints();// add points in n_pts to forw_pts, ids and track_cnt
        ROS_DEBUG("selectFeature costs: %fms", t_a.toc());
    }
    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    cur_img = forw_img;
    cur_pts = forw_pts;
    // un distort points for elements in cur_pts, make pair of their id and point and save to cur_un_pts_map, if we can find their corresponding in prev
    // frame, compute thier velocity using the diferrence of undistored coordinate in two frames and deltatime, else set their velocity to 0
    undistortedPoints();
    prev_time = cur_time;
}

void FeatureTracker::rejectWithF()
{
    if (forw_pts.size() >= 8)
    {
        ROS_DEBUG("FM ransac begins");
        TicToc t_f;
        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_forw_pts(forw_pts.size());
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            Eigen::Vector3d tmp_p;
            // Lifts a point from the image plane to its projective ray
            m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }
        // undistort cur_pts and forw_pts and put the points into un_cur_pts and un_forw_pts

        vector<uchar> status;
        // find fundamental matrix from a set of corresponding 2D points
        cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        int size_a = cur_pts.size();
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, forw_pts.size(), 1.0 * forw_pts.size() / size_a);
        ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}

bool FeatureTracker::updateID(unsigned int i)
{
    if (i < ids.size())
    {
        if (ids[i] == -1)
            ids[i] = n_id++;
        return true;
    }
    else
        return false;
}

void FeatureTracker::readIntrinsicParameter(const string &calib_file)
{
    ROS_INFO("reading paramerter of camera %s", calib_file.c_str());
    m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}

void FeatureTracker::showUndistortion(const string &name)
{
    cv::Mat undistortedImg(ROW + 600, COL + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < COL; i++)
        for (int j = 0; j < ROW; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            //printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + COL / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + ROW / 2;
        pp.at<float>(2, 0) = 1.0;
        //cout << trackerData[0].K << endl;
        //printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        //printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < ROW + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < COL + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            //ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    cv::imshow(name, undistortedImg);
    cv::waitKey(0);
}

void FeatureTracker::undistortedPoints()
{
    cur_un_pts.clear();
    cur_un_pts_map.clear();
    //cv::undistortPoints(cur_pts, un_pts, K, cv::Mat());
    for (unsigned int i = 0; i < cur_pts.size(); i++)
    {
        Eigen::Vector2d a(cur_pts[i].x, cur_pts[i].y);
        Eigen::Vector3d b;
        m_camera->liftProjective(a, b);
        cur_un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));// cur_un_pts stores the undistorted points
        cur_un_pts_map.insert(make_pair(ids[i], cv::Point2f(b.x() / b.z(), b.y() / b.z())));
        // cur_un_pts_map stores the pair of point id and undistored point
        //printf("cur pts id %d %f %f", ids[i], cur_un_pts[i].x, cur_un_pts[i].y);
    }
    // caculate points velocity
    if (!prev_un_pts_map.empty())// if the vector of prev points is not empty
    {
        double dt = cur_time - prev_time;
        pts_velocity.clear();
        for (unsigned int i = 0; i < cur_un_pts.size(); i++)// for every point in the vector cur_un_pts
        {
            // if the id is not -1. I think it means that the point is not trcked by the optical flow but was extracted using the mask,so it doesn't
            // have a corresponding point in the prev image
            if (ids[i] != -1)
            {
                std::map<int, cv::Point2f>::iterator it;
                it = prev_un_pts_map.find(ids[i]);// use the id to find corresponding in prev_un_pts_map
                if (it != prev_un_pts_map.end())
                {
                    double v_x = (cur_un_pts[i].x - it->second.x) / dt;
                    double v_y = (cur_un_pts[i].y - it->second.y) / dt;
                    pts_velocity.push_back(cv::Point2f(v_x, v_y));
                }
                else// if we can not find corresponding in the map prev_un_pts_map, set the velocity to 0
                    pts_velocity.push_back(cv::Point2f(0, 0));
            }
            else// if the point is extracted using the mask not tracked in the optical flow, set the velocity to 0
            {
                pts_velocity.push_back(cv::Point2f(0, 0));
            }
        }
    }
    else// if the map prev_un_pts_map is empty, set velocity for all points in cur_pts tobe 0
    {
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    prev_un_pts_map = cur_un_pts_map;
}
