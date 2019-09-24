#include "initial_sfm.h"

GlobalSFM::GlobalSFM(){}

/*
 * denote the 3D point as C, 2D pixel as x, pose as P, we have
 * x = P*C   use cross multiply, we can get [x]x(PC) = 0
 * | 0  -1  y |   |P(row0)C|
 * | 1   0  -x| * |P(row1)C| = 0
 * |-y   x  0 |   |P(row2)C|
 * -P(row1)C+yP(row2)C = 0   (1)
 * P(row0)C-xP(row2)C = 0    (2)
 * -yP(row0)C+xP(row1)C = 0  (3)
 * (3) can be expressed using (1) and (2)
 */
void GlobalSFM::triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
						Vector2d &point0, Vector2d &point1, Vector3d &point_3d)
{
	Matrix4d design_matrix = Matrix4d::Zero();
	design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
	design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
	design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
	design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);
	Vector4d triangulated_point;
	triangulated_point =
		      design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
	point_3d(0) = triangulated_point(0) / triangulated_point(3);
	point_3d(1) = triangulated_point(1) / triangulated_point(3);
	point_3d(2) = triangulated_point(2) / triangulated_point(3);
}

/*
 * 1. for all features observing by frames, if the feature is observed by frame i, put the corresponding 3D and 2D points in vector
 * 2. if the size of the vector is smaller than 15, warn the user that,feature tracking is unstable now and the device should slow down, if the size is
 *    smaller than 10, just return false
 * 3. solve pnp use the initial r and t, if not sucess return false, else reset r t using the result and return true
 */
bool GlobalSFM::solveFrameByPnP(Matrix3d &R_initial, Vector3d &P_initial, int i,
								vector<SFMFeature> &sfm_f)
{
	vector<cv::Point2f> pts_2_vector;
	vector<cv::Point3f> pts_3_vector;
	for (int j = 0; j < feature_num; j++)
	{
		if (sfm_f[j].state != true)// if the state of the j th feature is not true, continue to process next feature
			continue;
		Vector2d point2d;
		for (int k = 0; k < (int)sfm_f[j].observation.size(); k++) // every feature has k observations 
		{
			if (sfm_f[j].observation[k].first == i)//this means that the j th feature is observed by the i th frame
			{
				Vector2d img_pts = sfm_f[j].observation[k].second;
				cv::Point2f pts_2(img_pts(0), img_pts(1));// get the pixel coordinate of the j th feature point in the i th frame
				pts_2_vector.push_back(pts_2);
				cv::Point3f pts_3(sfm_f[j].position[0], sfm_f[j].position[1], sfm_f[j].position[2]);// get the 3D position of the j th feature point
				pts_3_vector.push_back(pts_3);
				break;
			}
		}
	}
	if (int(pts_2_vector.size()) < 15)// if here are less than 15 feature points among all feature points observed by the i th frame
	{
		printf("unstable features tracking, please slowly move you device!\n");
		if (int(pts_2_vector.size()) < 10)// if here are less than 10 feature points among all feature points observed by the i th frame, return false
			return false;
	}
	cv::Mat r, rvec, t, D, tmp_r;
	cv::eigen2cv(R_initial, tmp_r);
	cv::Rodrigues(tmp_r, rvec);
	cv::eigen2cv(P_initial, t);
	cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
	bool pnp_succ;
	/** @brief Finds an object pose from 3D-2D point correspondences.

	@param objectPoints Array of object points in the object coordinate space, Nx3 1-channel or
	1xN/Nx1 3-channel, where N is the number of points. vector\<Point3f\> can be also passed here.
	@param imagePoints Array of corresponding image points, Nx2 1-channel or 1xN/Nx1 2-channel,
	where N is the number of points. vector\<Point2f\> can be also passed here.
	@param cameraMatrix Input camera matrix \f$A = \vecthreethree{fx}{0}{cx}{0}{fy}{cy}{0}{0}{1}\f$ .
	@param distCoeffs Input vector of distortion coefficients
	\f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6 [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\f$ of
	4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are
	assumed.
	@param rvec Output rotation vector (see @ref Rodrigues ) that, together with tvec , brings points from
	the model coordinate system to the camera coordinate system.
	@param tvec Output translation vector.
	@param useExtrinsicGuess Parameter used for #SOLVEPNP_ITERATIVE. If true (1), the function uses
	the provided rvec and tvec values as initial approximations of the rotation and translation
	vectors, respectively, and further optimizes them.
	@param flags Method for solving a PnP problem:
	-   **SOLVEPNP_ITERATIVE** Iterative method is based on Levenberg-Marquardt optimization. In
	this case the function finds such a pose that minimizes reprojection error, that is the sum
	of squared distances between the observed projections imagePoints and the projected (using
	projectPoints ) objectPoints .
	-   **SOLVEPNP_P3P** Method is based on the paper of X.S. Gao, X.-R. Hou, J. Tang, H.-F. Chang
	"Complete Solution Classification for the Perspective-Three-Point Problem" (@cite gao2003complete).
	In this case the function requires exactly four object and image points.
	-   **SOLVEPNP_AP3P** Method is based on the paper of T. Ke, S. Roumeliotis
	"An Efficient Algebraic Solution to the Perspective-Three-Point Problem" (@cite Ke17).
	In this case the function requires exactly four object and image points.
	-   **SOLVEPNP_EPNP** Method has been introduced by F.Moreno-Noguer, V.Lepetit and P.Fua in the
	paper "EPnP: Efficient Perspective-n-Point Camera Pose Estimation" (@cite lepetit2009epnp).
	-   **SOLVEPNP_DLS** Method is based on the paper of Joel A. Hesch and Stergios I. Roumeliotis.
	"A Direct Least-Squares (DLS) Method for PnP" (@cite hesch2011direct).
	-   **SOLVEPNP_UPNP** Method is based on the paper of A.Penate-Sanchez, J.Andrade-Cetto,
	F.Moreno-Noguer. "Exhaustive Linearization for Robust Camera Pose and Focal Length
	Estimation" (@cite penate2013exhaustive). In this case the function also estimates the parameters \f$f_x\f$ and \f$f_y\f$
	assuming that both have the same value. Then the cameraMatrix is updated with the estimated
	focal length.
	-   **SOLVEPNP_AP3P** Method is based on the paper of Tong Ke and Stergios I. Roumeliotis.
	"An Efficient Algebraic Solution to the Perspective-Three-Point Problem". In this case the
	function requires exactly four object and image points.
	
	The function estimates the object pose given a set of object points, their corresponding image
	projections, as well as the camera matrix and the distortion coefficients.
	
	@note
	   -   An example of how to use solvePnP for planar augmented reality can be found at
	        opencv_source_code/samples/python/plane_ar.py
	   -   If you are using Python:
	        - Numpy array slices won't work as input because solvePnP requires contiguous
	        arrays (enforced by the assertion using cv::Mat::checkVector() around line 55 of
	        modules/calib3d/src/solvepnp.cpp version 2.4.9)
	        - The P3P algorithm requires image points to be in an array of shape (N,1,2) due
	        to its calling of cv::undistortPoints (around line 75 of modules/calib3d/src/solvepnp.cpp version 2.4.9)
	        which requires 2-channel information.
	        - Thus, given some data D = np.array(...) where D.shape = (N,M), in order to use a subset of
	        it as, e.g., imagePoints, one must effectively copy it into a new array: imagePoints =
	        np.ascontiguousarray(D[:,:2]).reshape((N,1,2))
	   -   The methods **SOLVEPNP_DLS** and **SOLVEPNP_UPNP** cannot be used as the current implementations are
	       unstable and sometimes give completly wrong results. If you pass one of these two
	       flags, **SOLVEPNP_EPNP** method will be used instead.
	   -   The minimum number of points is 4. In the case of **SOLVEPNP_P3P** and **SOLVEPNP_AP3P**
	       methods, it is required to use exactly 4 points (the first 3 points are used to estimate all the solutions
	       of the P3P problem, the last one is used to retain the best solution that minimizes the reprojection error).
	 */
	// rvec and t is the output rotation and translation vector
	pnp_succ = cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1);
	if(!pnp_succ)
	{
		return false;// if the pnp solver returns false, return false
	}
	cv::Rodrigues(rvec, r);
	//cout << "r " << endl << r << endl;
	MatrixXd R_pnp;
	cv::cv2eigen(r, R_pnp);
	MatrixXd T_pnp;
	cv::cv2eigen(t, T_pnp);
	R_initial = R_pnp;
	P_initial = T_pnp;
	return true;

}

/*
 * 1. check frame0 is not equal to frame1
 * 2. for all features
 *    2.1 for all frames observing the certain feature, check whether frame0 and frame1 is observing that feature
 *    2.2 if the feature is observed by both frame0 and frame1, using the pose of frame0 and frame1 to triangulate the certain feature, set 
 *        the state of the feature to true and set the position of the feature
 *        
 */
void GlobalSFM::triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4> &Pose0, 
									 int frame1, Eigen::Matrix<double, 3, 4> &Pose1,
									 vector<SFMFeature> &sfm_f)
{
	assert(frame0 != frame1);// frame0 shouldn't equal to frame1
	for (int j = 0; j < feature_num; j++)// for all features
	{
		if (sfm_f[j].state == true)// if the state of the feature is true, continue to process next feature
			continue;
		bool has_0 = false, has_1 = false;
		Vector2d point0;
		Vector2d point1;
		// for all frames observing the j th feature, check whether frame0 and frame1 is observing that feature
		for (int k = 0; k < (int)sfm_f[j].observation.size(); k++)
		{
			if (sfm_f[j].observation[k].first == frame0)
			{
				point0 = sfm_f[j].observation[k].second;
				has_0 = true;
			}
			if (sfm_f[j].observation[k].first == frame1)
			{
				point1 = sfm_f[j].observation[k].second;
				has_1 = true;
			}
		}
		if (has_0 && has_1)// if the j th feature was observed by both frame0 and frame1
		{
			Vector3d point_3d;
			triangulatePoint(Pose0, Pose1, point0, point1, point_3d);//using the pose of frame0 and frame1 to get the 3D position of the j th feature
			sfm_f[j].state = true;// set the state of the j th feature to true
			sfm_f[j].position[0] = point_3d(0);
			sfm_f[j].position[1] = point_3d(1);
			sfm_f[j].position[2] = point_3d(2);
			//cout << "trangulated : " << frame1 << "  3d point : "  << j << "  " << point_3d.transpose() << endl;
		}							  
	}
}

// 	 q w_R_cam t w_R_cam
//  c_rotation cam_R_w 
//  c_translation cam_R_w
// relative_q[i][j]  j_q_i
// relative_t[i][j]  j_t_ji  (j < i)
/*
 * 1. set the rotation and translation of the lth frame tobe 0, set the rotation and translation between the frame_num-1 and the lth frame tobe
 *    relative_R and relative_T
 * 2. initialize some arrays of vectors whose size is frame num to represent pose
 * 3. if frame i is in the range of [l, frame_num-1), if i>l use the pose of i-1 frame to initialize the pose of i frame, solve pnp to get the
 *    pose of the i th frame; then for frame i in range [l, frame_num-1), triangulate feature points between frame i and frame_num-1; then for 
 *    frame i in range [l+1, frame_num-1), triangulate feature points between frame l and frame i
 * 4. if frame i is in the range [l-1, 0], use the pose of i+1 frame to initialize the pose of i th frame, solve pnp to get the pose of the 
 *    i th frame, then triangulate feature points between frame i and frame l
 * 5. for all features, if the state of the feature is true, continue to process next feature, if it is observed by no less than 2 frames, get 
 *    the first frame observed the feature as frame 0 and the last frame observedthe feature as frame 1, use frame 0 and frame 1 to tiangulate
 *    the feature point, then set the feature's state to true and set the position of the feature
 * 6. for all frames, set their pose as the param of the optimization problem, set the rotation of the l th frame to be constant; set the 
 *    translation of the l th and frame_num-1 frame to be constant
 * 7. for all features
 *    7.1 if the state of the feature is not true, continue to process next feature
 *    7.2 for all frames observing the i th feature, use the 3D position of the i th feature, the pose of the l th frame and the pixel coordinate 
 *        of the l th frame to construct a residual
 * 8. for every frame , reset the rotation and translation, for every feature, if the state of the feature is true, reset the position of sfm_tracked_points
 */
bool GlobalSFM::construct(int frame_num, Quaterniond* q, Vector3d* T, int l,
			  const Matrix3d relative_R, const Vector3d relative_T,
			  vector<SFMFeature> &sfm_f, map<int, Vector3d> &sfm_tracked_points)
{
	feature_num = sfm_f.size();// get the number of features
	//cout << "set 0 and " << l << " as known " << endl;
	// have relative_r relative_t
	// intial two view
	// set the rotation and translation of the lth frame tobe 0, set the rotation and translation between the frame_num -1 frame and the lth frame 
	// to be relative_r and relative_t
	q[l].w() = 1;
	q[l].x() = 0;
	q[l].y() = 0;
	q[l].z() = 0;
	T[l].setZero();
	q[frame_num - 1] = q[l] * Quaterniond(relative_R);
	T[frame_num - 1] = relative_T;
	//cout << "init q_l " << q[l].w() << " " << q[l].vec().transpose() << endl;
	//cout << "init t_l " << T[l].transpose() << endl;

	//rotate to cam frame
	// initialize some arrays of vectors whose size is frame num to represent pose
	Matrix3d c_Rotation[frame_num];
	Vector3d c_Translation[frame_num];
	Quaterniond c_Quat[frame_num];
	double c_rotation[frame_num][4];
	double c_translation[frame_num][3];
	Eigen::Matrix<double, 3, 4> Pose[frame_num];

	c_Quat[l] = q[l].inverse();
	c_Rotation[l] = c_Quat[l].toRotationMatrix();
	c_Translation[l] = -1 * (c_Rotation[l] * T[l]);
	Pose[l].block<3, 3>(0, 0) = c_Rotation[l];
	Pose[l].block<3, 1>(0, 3) = c_Translation[l];

	c_Quat[frame_num - 1] = q[frame_num - 1].inverse();
	c_Rotation[frame_num - 1] = c_Quat[frame_num - 1].toRotationMatrix();
	c_Translation[frame_num - 1] = -1 * (c_Rotation[frame_num - 1] * T[frame_num - 1]);
	Pose[frame_num - 1].block<3, 3>(0, 0) = c_Rotation[frame_num - 1];
	Pose[frame_num - 1].block<3, 1>(0, 3) = c_Translation[frame_num - 1];


	/*
	 * 1. if frame i is in the range of [l, frame_num-1), if i>l use the pose of i-1 frame to initialize the pose of i frame, solve pnp to get the
	 *    pose of the i th frame; then for frame i in range [l, frame_num-1), triangulate feature points between frame i and frame_num-1; then for 
	 *    frame i in range [l+1, frame_num-1), triangulate feature points between frame l and frame i
	 * 2. if frame i is in the range [l-1, 0], use the pose of i+1 frame to initialize the pose of i th frame, solve pnp to get the pose of the 
	 *    i th frame, then triangulate feature points betweenframe i and frame l
	 */
	//1: trangulate between l ----- frame_num - 1
	//2: solve pnp l + 1; trangulate l + 1 ------- frame_num - 1; 
	for (int i = l; i < frame_num - 1 ; i++)
	{
		// solve pnp
		if (i > l)// for frame l+1, ..., frame_num-1, use the pose of last frame to initialize the current frame, use the initial guess to solvepnp
		{
			Matrix3d R_initial = c_Rotation[i - 1];
			Vector3d P_initial = c_Translation[i - 1];
			if(!solveFrameByPnP(R_initial, P_initial, i, sfm_f))
				return false;
			c_Rotation[i] = R_initial;
			c_Translation[i] = P_initial;
			c_Quat[i] = c_Rotation[i];
			Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
			Pose[i].block<3, 1>(0, 3) = c_Translation[i];
		}

		// triangulate point based on the solve pnp result
		// use the result of the pnp to triangulate between i th frame and frame_num-1 frame
		triangulateTwoFrames(i, Pose[i], frame_num - 1, Pose[frame_num - 1], sfm_f);
	}
	//3: triangulate l-----l+1 l+2 ... frame_num -2
	// for frame l+1, ..., frame_num-2, triangulate these frames with frame l
	for (int i = l + 1; i < frame_num - 1; i++)
		triangulateTwoFrames(l, Pose[l], i, Pose[i], sfm_f);
	//4: solve pnp l-1; triangulate l-1 ----- l
	//             l-2              l-2 ----- l
	// for frame l-1, ..., 0, use the pose of frame i+1 to initialize frame i, if solve pnp returns true, triangulate points using frame l and frame i 
	for (int i = l - 1; i >= 0; i--)
	{
		//solve pnp
		Matrix3d R_initial = c_Rotation[i + 1];
		Vector3d P_initial = c_Translation[i + 1];
		if(!solveFrameByPnP(R_initial, P_initial, i, sfm_f))
			return false;
		c_Rotation[i] = R_initial;
		c_Translation[i] = P_initial;
		c_Quat[i] = c_Rotation[i];
		Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
		Pose[i].block<3, 1>(0, 3) = c_Translation[i];
		//triangulate
		triangulateTwoFrames(i, Pose[i], l, Pose[l], sfm_f);
	}
	//5: triangulate all other points
	// for all features, if the state of the feature is true, continue to process next feature, if it is observed by no less than 2 frames, get 
	// the first frame observed the feature as frame 0 and the last frame observedthe feature as frame 1, use frame 0 and frame 1 to tiangulate
	// the feature point, then set the feature's state to true and set the position of the feature
	for (int j = 0; j < feature_num; j++)
	{
		if (sfm_f[j].state == true)
			continue;
		if ((int)sfm_f[j].observation.size() >= 2)
		{
			Vector2d point0, point1;
			int frame_0 = sfm_f[j].observation[0].first;
			point0 = sfm_f[j].observation[0].second;
			int frame_1 = sfm_f[j].observation.back().first;
			point1 = sfm_f[j].observation.back().second;
			Vector3d point_3d;
			triangulatePoint(Pose[frame_0], Pose[frame_1], point0, point1, point_3d);
			sfm_f[j].state = true;
			sfm_f[j].position[0] = point_3d(0);
			sfm_f[j].position[1] = point_3d(1);
			sfm_f[j].position[2] = point_3d(2);
			//cout << "trangulated : " << frame_0 << " " << frame_1 << "  3d point : "  << j << "  " << point_3d.transpose() << endl;
		}		
	}

/*
	for (int i = 0; i < frame_num; i++)
	{
		q[i] = c_Rotation[i].transpose(); 
		cout << "solvePnP  q" << " i " << i <<"  " <<q[i].w() << "  " << q[i].vec().transpose() << endl;
	}
	for (int i = 0; i < frame_num; i++)
	{
		Vector3d t_tmp;
		t_tmp = -1 * (q[i] * c_Translation[i]);
		cout << "solvePnP  t" << " i " << i <<"  " << t_tmp.x() <<"  "<< t_tmp.y() <<"  "<< t_tmp.z() << endl;
	}
*/
	//full BA
	ceres::Problem problem;
	ceres::LocalParameterization* local_parameterization = new ceres::QuaternionParameterization();
	//cout << " begin full BA " << endl;
	for (int i = 0; i < frame_num; i++)
	{
		//double array for ceres
		c_translation[i][0] = c_Translation[i].x();
		c_translation[i][1] = c_Translation[i].y();
		c_translation[i][2] = c_Translation[i].z();
		c_rotation[i][0] = c_Quat[i].w();
		c_rotation[i][1] = c_Quat[i].x();
		c_rotation[i][2] = c_Quat[i].y();
		c_rotation[i][3] = c_Quat[i].z();
		problem.AddParameterBlock(c_rotation[i], 4, local_parameterization);
		problem.AddParameterBlock(c_translation[i], 3);
		if (i == l)// set the rotation of the l th frame to be constant
		{
			problem.SetParameterBlockConstant(c_rotation[i]);
		}
		if (i == l || i == frame_num - 1)// set the translation of the l th and frame_num-1 frame to be constant
		{
			problem.SetParameterBlockConstant(c_translation[i]);
		}
	}

	for (int i = 0; i < feature_num; i++)// for all features
	{
		if (sfm_f[i].state != true)// if the state of the feature is not true, continue to process next feature
			continue;
		for (int j = 0; j < int(sfm_f[i].observation.size()); j++) // for all frames observing the i th feature
		{
			int l = sfm_f[i].observation[j].first;
			ceres::CostFunction* cost_function = ReprojectionError3D::Create(
												sfm_f[i].observation[j].second.x(),
												sfm_f[i].observation[j].second.y());

    		problem.AddResidualBlock(cost_function, NULL, c_rotation[l], c_translation[l], 
    								sfm_f[i].position);	 
			// use the 3D position of the i th feature, the pose of the l th frame and the pixel coordinate of the l th frame to construct a residual
		}

	}
	// use ceres to solve the optimization problem
	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	//options.minimizer_progress_to_stdout = true;
	options.max_solver_time_in_seconds = 0.2;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	//std::cout << summary.BriefReport() << "\n";
	if (summary.termination_type == ceres::CONVERGENCE || summary.final_cost < 5e-03)
	{
		//cout << "vision only BA converge" << endl;
	}
	else
	{
		//cout << "vision only BA not converge " << endl;
		return false;
	}
	for (int i = 0; i < frame_num; i++)// for every frame , reset the rotation and translation
	{
		q[i].w() = c_rotation[i][0]; 
		q[i].x() = c_rotation[i][1]; 
		q[i].y() = c_rotation[i][2]; 
		q[i].z() = c_rotation[i][3]; 
		q[i] = q[i].inverse();
		//cout << "final  q" << " i " << i <<"  " <<q[i].w() << "  " << q[i].vec().transpose() << endl;
	}
	for (int i = 0; i < frame_num; i++)
	{

		T[i] = -1 * (q[i] * Vector3d(c_translation[i][0], c_translation[i][1], c_translation[i][2]));
		//cout << "final  t" << " i " << i <<"  " << T[i](0) <<"  "<< T[i](1) <<"  "<< T[i](2) << endl;
	}
	// for every feature, if the state of the feature is true, reset the position of sfm_tracked_points
	for (int i = 0; i < (int)sfm_f.size(); i++)
	{
		if(sfm_f[i].state)
			sfm_tracked_points[sfm_f[i].id] = Vector3d(sfm_f[i].position[0], sfm_f[i].position[1], sfm_f[i].position[2]);
	}
	return true;

}

