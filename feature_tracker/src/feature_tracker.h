#pragma once

#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"

#include "parameters.h"
#include "tic_toc.h"

using namespace std;
using namespace camodocal;
using namespace Eigen;

// check whether the point is in the rect (BORDER_SIZE:(ROW - BORDER_SIZE), BORDER_SIZE:(COL - BORDER_SIZE))
bool inBorder(const cv::Point2f &pt);

/*
 * every element in v has a corresponding stat in status, if status is ok, put the element in the new vector, else delete the element
 */
void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);
// use the element in vector status corresponding to the element in vector v to determine whether the element will be kept in vector v
void reduceVector(vector<int> &v, vector<uchar> status);

class FeatureTracker
{
  public:
    FeatureTracker();

    /*
     * brief:
     * this function find corresponding points between two adjacent frames(also detect new features for the new image), and compute velocity in the undistorted
     * camera coordinate
     * 
     * In detail: 
     * 1. using img from the imput to assign forw_img
     * 2. using Lucas-Kanade optical flow in pyramids to find corresponding points in cur_img and prev_img, use the status of LK to reduce some elements
     *    in some corresponding vectors
     * 3. if PUB_THIS_FRAME is true
     *    3.1 compute fundamental matrix using a set of corresponding points, use the status of fundamental computation to delete some elements in the vector
     *    3.2 construct a mask using forw_pts
     *    3.3 if forw_pts's size is smaller than MAX_CNT, detect (MAX_CNT-forw_pts.size()) features in forw_img(this step take mask constructed in 3.2
     *        into consideration), add the detected points to forw_pts, ids and track_cnt
     * 4. use cur_un_pts_map and prev_un_pts_map to compute velocity for feature points, if we can not find valid velocity for a point, use (0, 0) as its velocity
     */
    void readImage(const cv::Mat &_img,double _cur_time);

    /*
     * brief: 
     * compute a mask, so latter when detecting new features we can restrain features near exsiting features
     * 
     * In details: 
     * 1. if the camera is a fisheye camera, use the fisheye_mask as the initial mask, else use a all 255-value image as the initial mask
     * 2. construct a vector<pair<int, pair<cv::Point2f, int>>> object cnt_pts_id, every element is a trackcount, its corresponding forward
     *    points and the point id, for every point in vector forw_pts, put its infomation to cnts_pts_id
     * 3. for every element in vector cnt_pts_id, if the mask at the feature point in the element is 255, push the infomation separately 
     *    into empty vectors forw_pts, ids, track_cnt; then draw a circle whose center is the feature point and radius is MIN_DIST, then
     *    fill the circle with 0-value
     */
    void setMask();

    // for p in vector n_pts, push back p to forw_pts, push back -1 to ids, and push_back 1 to track_cnt
    void addPoints();

    /*
     * brief:
     * update id for a point
     * 
     * in detail:
     * 1. if i is smaller than ids.size(), I think it means i is an index of vector ids, if ids[i] equals -1, assign ids[i] using n_id++ and return true
     * 2. if i is no smaller than ids.size(), return false
     */
    bool updateID(unsigned int i);

    /*
     * brief:
     * read the camera intrinsic from the calib file
     */
    void readIntrinsicParameter(const string &calib_file);

    /*
     * brief:
     * undistort cur_img and show
     * 
     * in detail:
     * 1. for every coordinate in the range (0:ROW, 0:COL), push back the origin coordinate to vector distortedp, and push back the undistorted
     *    coordinate to vector undistortedp
     * 2. for every element in the vector undistortedp, use the corresponding coordinate from distortedp and undistortedp to reproject cur_img
     *    to undistortedImg
     * 3. show undistortedImg
     */
    void showUndistortion(const string &name);

    /*
     * brief:
     * compute F matrix using un_pts from two consecutive img, then rejct some pair of corresponding points using the F matrix
     * 
     * in details:
     * 1. undistort cur_pts and forw_pts and push the undistorted points separately into un_cur_pts and un_forw_pts
     * 2. use un_cur_pts and un_forw_pts to compute fundamental matrix, use the status of this step to reduce some corresponding vectors
     */
    void rejectWithF();

    /*
     * brief:
     * undistort cur_pts and use cur_un_pts_map and prev_un_pts_map to compute velocity for feature points, if we can not find valid 
     * velocity for a point, use (0, 0) as its velocity
     * 
     * in detail:
     * 1. for every element in the vector cur_pts, compute the undistorted camera coordinate, put the undistorted camera coordinate into 
     *    the empty vector cur_un_pts and put the pair of corresponding index and the undistorted camera coordinate into map cur_un_pts_map
     * 2. if the map prev_un_pts_map is not empty
     *    2.1 compute dt using cur_time subtract prev_time, clear the vector<cv::Point2f> object pts_velocity
     *    2.2 for every element in vector cur_un_pts
     *        2.2.1 if the corresponding index is not -1 (if the index is -1, it means the point is not tracked 
     *              between frames, instead it is directly detected from a single frame), use the index to get the corresponding undistorted camera 
     *              coordinate in prev_un_pts_map, if we can find the corresponding coordinate, use (diffrence_between_cur_undist_and_prev_dist/dt)
     *              to compute the velocity, put the velocity into vector pts_velocity; else, we can not find the corresponding undistorted coordinate 
     *              in pts_velocity, so just push back (0, 0) to vector pts_velocity
     *        2.2.2 else, the corresponding index is -1, just push back (0, 0) to vector pts_velocity
     * 3. else, the map prev_un_pts_map is empty, push back (size of cur_pts) (0, 0) to vector pts_velocity
     * 4. using cur_un_pts_map to assign prev_un_pts_map 
     */
    void undistortedPoints();

    cv::Mat mask;
    cv::Mat fisheye_mask;
    cv::Mat prev_img, cur_img, forw_img;
    vector<cv::Point2f> n_pts;
    vector<cv::Point2f> prev_pts, cur_pts, forw_pts;
    vector<cv::Point2f> prev_un_pts, cur_un_pts;
    vector<cv::Point2f> pts_velocity;
    vector<int> ids;// I think ids stores the inex of the feature among all features in the system instead of index in a single frame
    vector<int> track_cnt;
    map<int, cv::Point2f> cur_un_pts_map;
    map<int, cv::Point2f> prev_un_pts_map;
    camodocal::CameraPtr m_camera;
    double cur_time;
    double prev_time;

    static int n_id; 
    /*
     * n_id is a static so I think it do not belong to an object but belong to the class, that's why I infer ids stores index of the 
     * feature among all features in the system
     */
};
