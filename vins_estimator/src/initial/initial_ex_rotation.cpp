#include "initial_ex_rotation.h"

InitialEXRotation::InitialEXRotation(){
    frame_count = 0;
    Rc.push_back(Matrix3d::Identity());
    Rc_g.push_back(Matrix3d::Identity());
    Rimu.push_back(Matrix3d::Identity());
    ric = Matrix3d::Identity();
}

/*
 * for quaternion whose formulation is in order qw+i*qx+j*qy+k*qz, expressed in vector formulation as [qw qx qy qz]'
 * q1*q2 = [q1]L*q2      q1*q2 = [q2]R*q1
 *        | qw  -qx  -qy  -qz |          | qw  -qx  -qy  -qz |
 *        | qx   qw  -qz   qy |          | qx   qw   qz  -qy |
 * [q]L = | qy   qz   qw  -qx |   [q]R = | qy  -qz   qw   qx |
 *        | qz  -qy   qx   qw |          | qz   qy  -qx   qw |
 *                        |  0   -az   ay |
 * skew-symmetric  [a]x = |  az   0   -ax |
 *                        | -ay   ax   0  |
 */
bool InitialEXRotation::CalibrationExRotation(vector<pair<Vector3d, Vector3d>> corres, Quaterniond delta_q_imu, Matrix3d &calib_ric_result)
{
    // R_bi_to_bj = R_cj_to_bj*R_ci_to_cj*R_bi_to_ci = R_c_to_b*R_ci_to_cj*R_b_to_c ==> R_b_to_c*R_bi_to_bj = R_ci_to_cj*R_b_to_c
    // R_ci_to_cj = R_bj_to_cj*R_bi_to_bj*R_ci_to_bi = R_b_to_c*R_bi_to_bj*R_c_to_b ==> R_c_to_b*R_ci_to_cj = R_bi_to_bj*R_c_to_b
    // [q_bi_to_bj]R * q_b_to_c = [q_ci_to_cj]L * q_b_to_c ==> ([q_bi_to_bj]R - [q_ci_to_cj]L) * q_b_to_c = 0
    // [q_ci_to_cj]R * q_c_to_b = [q_bi_to_bj]L * q_c_to_b ==> ([q_ci_to_cj]R - [q_bi_to_bj]L) * q_c_to_b = 0
    frame_count++;
    // solveRelativeR use the corresponding two vector to compute the essential matrix and then decompose E matrix get 2 r and 2 t, use the percentage  
    // of valid 3d points to choose the best R
    Rc.push_back(solveRelativeR(corres));// rotation computed from the camera data
    Rimu.push_back(delta_q_imu.toRotationMatrix());// rotation computed from the imu data
    // use the imu computed rotation and relative rotation between imu and camera to compute the camera rotation
    Rc_g.push_back(ric.inverse() * delta_q_imu * ric);

    Eigen::MatrixXd A(frame_count * 4, 4);// every pair of transform can provide a equation A*ric=0
    A.setZero();
    int sum_ok = 0;
    for (int i = 1; i <= frame_count; i++)
    {
        Quaterniond r1(Rc[i]);// rotation computed from the camera data
        Quaterniond r2(Rc_g[i]);//camera rotation computed according to relative rotation between imu rotation and (camera and imu rotation)

        double angular_distance = 180 / M_PI * r1.angularDistance(r2);
        ROS_DEBUG(
            "%d %f", i, angular_distance);

        // the bigger the angular_distance is, the smaller huber will be
        double huber = angular_distance > 5.0 ? 5.0 / angular_distance : 1.0;
        ++sum_ok;
        Matrix4d L, R;

        double w = Quaterniond(Rc[i]).w();// real part
        Vector3d q = Quaterniond(Rc[i]).vec();// imaginary part
        L.block<3, 3>(0, 0) = w * Matrix3d::Identity() + Utility::skewSymmetric(q);
        L.block<3, 1>(0, 3) = q;
        L.block<1, 3>(3, 0) = -q.transpose();
        L(3, 3) = w;

        Quaterniond R_ij(Rimu[i]);
        w = R_ij.w();
        q = R_ij.vec();
        R.block<3, 3>(0, 0) = w * Matrix3d::Identity() - Utility::skewSymmetric(q);
        R.block<3, 1>(0, 3) = q;
        R.block<1, 3>(3, 0) = -q.transpose();
        R(3, 3) = w;

        A.block<4, 4>((i - 1) * 4, 0) = huber * (L - R);
    }

    JacobiSVD<MatrixXd> svd(A, ComputeFullU | ComputeFullV);
    Matrix<double, 4, 1> x = svd.matrixV().col(3);
    Quaterniond estimated_R(x);
    ric = estimated_R.toRotationMatrix().inverse();
    //cout << svd.singularValues().transpose() << endl;
    //cout << ric << endl;
    Vector3d ric_cov;
    ric_cov = svd.singularValues().tail<3>();
    if (frame_count >= WINDOW_SIZE && ric_cov(1) > 0.25)
    {
        calib_ric_result = ric;
        return true;
    }
    else
        return false;
}

Matrix3d InitialEXRotation::solveRelativeR(const vector<pair<Vector3d, Vector3d>> &corres)
{
    if (corres.size() >= 9)
    {
        vector<cv::Point2f> ll, rr;
        for (int i = 0; i < int(corres.size()); i++)
        {
            ll.push_back(cv::Point2f(corres[i].first(0), corres[i].first(1)));
            rr.push_back(cv::Point2f(corres[i].second(0), corres[i].second(1)));
        }
        /*
         * @brief Calculates a fundamental matrix from the corresponding points in two images.
         *   
         * @param points1 Array of N points from the first image. The point coordinates should be
         * floating-point (single or double precision).
         * @param points2 Array of the second image points of the same size and format as points1 .
         * @param method Method for computing a fundamental matrix.
         * -   **CV_FM_7POINT** for a 7-point algorithm. \f$N = 7\f$
         * -   **CV_FM_8POINT** for an 8-point algorithm. \f$N \ge 8\f$
         * -   **CV_FM_RANSAC** for the RANSAC algorithm. \f$N \ge 8\f$
         * -   **CV_FM_LMEDS** for the LMedS algorithm. \f$N \ge 8\f$
         * @param param1 Parameter used for RANSAC. It is the maximum distance from a point to an epipolar
         * line in pixels, beyond which the point is considered an outlier and is not used for computing the
         * final fundamental matrix. It can be set to something like 1-3, depending on the accuracy of the
         * point localization, image resolution, and the image noise.
         * @param param2 Parameter used for the RANSAC or LMedS methods only. It specifies a desirable level
         * of confidence (probability) that the estimated matrix is correct.
         * @param mask
         * 
         * The epipolar geometry is described by the following equation:
         * 
         * \f[[p_2; 1]^T F [p_1; 1] = 0\f]
         * 
         * where \f$F\f$ is a fundamental matrix, \f$p_1\f$ and \f$p_2\f$ are corresponding points in the first and the
         * second images, respectively.
         * 
         * The function calculates the fundamental matrix using one of four methods listed above and returns
         * the found fundamental matrix. Normally just one matrix is found. But in case of the 7-point
         * algorithm, the function may return up to 3 solutions ( \f$9 \times 3\f$ matrix that stores all 3
         * matrices sequentially).
         * 
         * The calculated fundamental matrix may be passed further to computeCorrespondEpilines that finds the
         * epipolar lines corresponding to the specified points. It can also be passed to
         * stereoRectifyUncalibrated to compute the rectification transformation. :
         * @code
         *     // Example. Estimation of fundamental matrix using the RANSAC algorithm
         *     int point_count = 100;
         *     vector<Point2f> points1(point_count);
         *     vector<Point2f> points2(point_count);
         * 
         *     // initialize the points here ...
         *     for( int i = 0; i < point_count; i++ )
         *     {
         *         points1[i] = ...;
         *         points2[i] = ...;
         *     }
         * 
         *     Mat fundamental_matrix =
         *      findFundamentalMat(points1, points2, FM_RANSAC, 3, 0.99);
         * @endcode
         * 
         * CV_EXPORTS_W Mat findFundamentalMat( InputArray points1, InputArray points2,
         *                                      int method = FM_RANSAC,
         *                                      double param1 = 3., double param2 = 0.99,
         *                                      OutputArray mask = noArray() );
         */
        // how can the function findFundamentalMat return the essential matrix ???
        cv::Mat E = cv::findFundamentalMat(ll, rr);
        cv::Mat_<double> R1, R2, t1, t2;
        /*
         *               | 1       0               0      |
         * Rx(theta_x) = | 0  cos(theta_x)  -sin(theta_x) |
         *               | 0  sin(theta_x)   cos(theta_x) |
         * 
         *               |  cos(theta_y)  0  sin(theta_y) |
         * Ry(theta_y) = |       0        1        0      |     no matter what value theta is, the determinant of the three matrix is always 1
         *               | -sin(theta_y)  0  cos(theta_y) |
         * 
         *               | cos(theta_z)  -sin(theta_z)  0 |
         * Rz(theta_z) = | sin(theta_z)   cos(theta_z)  0 |
         *               |      0               0       1 |
         * 
         * the rotation matrix is the product of the above three matrix Rx,Ry,Rz whose determinant is 1, so the determinant of rotation matrix must 
         * be 1 and won't be -1
         */
        decomposeE(E, R1, R2, t1, t2);

        if (determinant(R1) + 1.0 < 1e-09)// the determinant  of rotation matrix must be 1 and won't be -1
        {
            E = -E;
            decomposeE(E, R1, R2, t1, t2);
        }
        // use the 4 pair of rt to triangulate corresponding pixel, choose the r matrix according to the percentage of valid 3d points
        double ratio1 = max(testTriangulation(ll, rr, R1, t1), testTriangulation(ll, rr, R1, t2));
        double ratio2 = max(testTriangulation(ll, rr, R2, t1), testTriangulation(ll, rr, R2, t2));
        cv::Mat_<double> ans_R_cv = ratio1 > ratio2 ? R1 : R2;

        Matrix3d ans_R_eigen;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                ans_R_eigen(j, i) = ans_R_cv(i, j);
        return ans_R_eigen;// return the choosen r matrix
    }
    // if the number of corresponding pixel is less than 9, just return identity 
    return Matrix3d::Identity();
}

// given the corresponding pixel points and reletive R t, triangulate the points, and return the percentage of the valid 
// 3d points(points with positive depth)
double InitialEXRotation::testTriangulation(const vector<cv::Point2f> &l,
                                          const vector<cv::Point2f> &r,
                                          cv::Mat_<double> R, cv::Mat_<double> t)
{
    cv::Mat pointcloud;
    cv::Matx34f P = cv::Matx34f(1, 0, 0, 0,
                                0, 1, 0, 0,
                                0, 0, 1, 0);
    cv::Matx34f P1 = cv::Matx34f(R(0, 0), R(0, 1), R(0, 2), t(0),
                                 R(1, 0), R(1, 1), R(1, 2), t(1),
                                 R(2, 0), R(2, 1), R(2, 2), t(2));
    /*
     * @brief Reconstructs points by triangulation.

     * @param projMatr1 3x4 projection matrix of the first camera.
     * @param projMatr2 3x4 projection matrix of the second camera.
     * @param projPoints1 2xN array of feature points in the first image. In case of c++ version it can
     * be also a vector of feature points or two-channel matrix of size 1xN or Nx1.
     * @param projPoints2 2xN array of corresponding points in the second image. In case of c++ version
     * it can be also a vector of feature points or two-channel matrix of size 1xN or Nx1.
     * @param points4D 4xN array of reconstructed points in homogeneous coordinates.
     * 
     * The function reconstructs 3-dimensional points (in homogeneous coordinates) by using their
     * observations with a stereo camera. Projections matrices can be obtained from stereoRectify.
     * 
     * @note
     *    Keep in mind that all input data should be of float type in order for this function to work.
     * 
     * @sa
     *    reprojectImageTo3D
     */
    /*
     * the cv::Mat pointcloud is the output of the function, it contains a series of point4D who is the reconstructed points in homogeneous coordinate
     */
    cv::triangulatePoints(P, P1, l, r, pointcloud);
    int front_count = 0;
    // pointcloud.cols is the number of the computed 3d points
    for (int i = 0; i < pointcloud.cols; i++)
    {
        double normal_factor = pointcloud.col(i).at<float>(3);// normal_factor is the scale of the vector

        cv::Mat_<double> p_3d_l = cv::Mat(P) * (pointcloud.col(i) / normal_factor);// normalize the 3d point in the l camera coordinate
        cv::Mat_<double> p_3d_r = cv::Mat(P1) * (pointcloud.col(i) / normal_factor);// normalize the 3d point in the r camera coordinate
        if (p_3d_l(2) > 0 && p_3d_r(2) > 0)// if the depth in both the l and r camera coordinate 
            front_count++;
        // if the point in both l and r coordinate has positive depth, add front_count by 1
    }
    ROS_DEBUG("MotionEstimator: %f", 1.0 * front_count / pointcloud.cols);
    return 1.0 * front_count / pointcloud.cols;
    // return the percentage of the valid(positive depth) 3d points
}

/*
 * E = K.transpose*F*K
 *     |  0 1 0 |      | 0 -1 0 |
 * Z = | -1 0 0 |  W = | 1  0 0 |
 *     |  0 0 0 |      | 0  0 1 |
 * E = [t]xR, we can get that E=SR, we can use S to express t
 * if the SVD decomposion of E is U*diag(1, 1, 0)*V', then E=SR can be written in two ways
 * E=U*diag(1, 1, 0)*V' = U*Z*W*V' = U*Z*U'*U*W*V' = (U*Z*U')*(U*W*V')
 * S = UZU'   R=UWV' or R=UW'V'   t=U3 = U(0, 0, 1)'
 */
void InitialEXRotation::decomposeE(cv::Mat E,
                                 cv::Mat_<double> &R1, cv::Mat_<double> &R2,
                                 cv::Mat_<double> &t1, cv::Mat_<double> &t2)
{
    cv::SVD svd(E, cv::SVD::MODIFY_A);
    cv::Matx33d W(0, -1, 0,
                  1, 0, 0,
                  0, 0, 1);
    cv::Matx33d Wt(0, 1, 0,
                   -1, 0, 0,
                   0, 0, 1);
    R1 = svd.u * cv::Mat(W) * svd.vt;
    R2 = svd.u * cv::Mat(Wt) * svd.vt;
    t1 = svd.u.col(2);
    t2 = -svd.u.col(2);
}
