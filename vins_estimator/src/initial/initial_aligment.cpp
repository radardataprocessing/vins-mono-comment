#include "initial_alignment.h"

/*
 * it is not easy to write the special notation in a IDE, so just use * to represents the multiplication for quaternion
 * the objective function is that, the orientation between two frames should equals the pre-integration of imu
 * estimate a delta_bw who would minimize sum of all k two-norm(q_bk+1_to_c0.inverse*q_bk_to_c0*gamma_bk+1_to_bk)
 * use ~~ to express approximately equal to 
 * gamma_bk+1_to_bk ~~ gamma_bk+1_to_bk_pre_integration_estimate*[1 1/2*J_gamma_about_bw*delta_bw].transpose()
 * the minimum of the above mentioned value is the unit quaternion, so the objective function can be written to 
 * q_bk+1_to_c0.inverse*q_bk_to_c0*gamma_bk+1_to_bk = [1 0].transpose()
 * q_bk+1_to_c0.inverse*q_bk_to_c0*gamma_bk+1_to_bk_pre_integration_estimate*[1 1/2*J_gamma_about_bw*delta_bw].transpose() = [1 0].transpose()
 * left multiply gamma_bk+1_to_bk_pre_integration_estimat.inverse()*q_bk_to_c0.inverse()*q_bk+1_to_c0 on both two sides of the equal sign, we can get
 * [1 1/2*J_gamma_about_bw*delta_bw].transpose() = gamma_bk+1_to_bk_pre_integration_estimat.inverse()*q_bk_to_c0.inverse()*q_bk+1_to_c0
 * just process the imaginary part, we get 
 * J_gamma_about_bw*delta_bw=2*(gamma_bk+1_to_bk_pre_integration_estimat.inverse()*q_bk_to_c0.inverse()*q_bk+1_to_c0).imaginary
 * so J_gamma_about_bw.transpose()*J_gamma_about_bw*delta_bw
 *   =2*J_gamma_about_bw.transpose()*(gamma_bk+1_to_bk_pre_integration_estimat.inverse()*q_bk_to_c0.inverse()*q_bk+1_to_c0).imaginary
 * 
 */

/**
 * Eigen::LLT performs a LL^T Cholesky decomposition of a symmetric, positive definite matrix A such that A=LL^*=U^*U, where L is lower triangular
 * While the Cholesky decomposition is particularly useful to solve selfadjoint problems like D^*Dx=b, for that purpose, we recommand the Cholesky
 * decomposition without square root which is more stable and even faster.Nevertheless, this standard Cholesky decomposition remains useful in many 
 * other situations like generalised eigen problems with hermitian matrices
 * Cholesky decompositions are not rank-revealing. This LLT decomposition is only stable on positive definite matrices, use LDLT instead for the 
 * semidefinite case.Also, do not use a Cholesky decomposition to determine whether a system of equations has a solution.  
 */

/**
 * 
 */
void solveGyroscopeBias(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs)
{
    Matrix3d A;
    Vector3d b;
    Vector3d delta_bg;
    A.setZero();
    b.setZero();
    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++)
    {
        frame_j = next(frame_i);// frame i is a frame in the map, frame j is frame i's adjacent frame 
        MatrixXd tmp_A(3, 3);
        tmp_A.setZero();
        VectorXd tmp_b(3);
        tmp_b.setZero();
        Eigen::Quaterniond q_ij(frame_i->second.R.transpose() * frame_j->second.R);// orientation between frame i and j
        /*
         * enum StateOrder         represents the position of that part in the state vector
         * {
         *     O_P = 0,
         *     O_R = 3,
         *     O_V = 6,
         *     O_BA = 9,
         *     O_BG = 12
         * };
         */
        tmp_A = frame_j->second.pre_integration->jacobian.template block<3, 3>(O_R, O_BG);
        tmp_b = 2 * (frame_j->second.pre_integration->delta_q.inverse() * q_ij).vec();//vec() is used to get the imaginary part of the quaternion
        A += tmp_A.transpose() * tmp_A;
        b += tmp_A.transpose() * tmp_b;

    }
    delta_bg = A.ldlt().solve(b); // the g in bg here means gyroscope
    ROS_WARN_STREAM("gyroscope bias initial calibration " << delta_bg.transpose());

    for (int i = 0; i <= WINDOW_SIZE; i++)
        Bgs[i] += delta_bg;

    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++)
    {
        frame_j = next(frame_i);
        frame_j->second.pre_integration->repropagate(Vector3d::Zero(), Bgs[0]);
    }
}


MatrixXd TangentBasis(Vector3d &g0)
{
    Vector3d b, c;
    Vector3d a = g0.normalized();
    Vector3d tmp(0, 0, 1);
    if(a == tmp)
        tmp << 1, 0, 0;
    // a.transpose()* tmp is a scalar whose value is a.norm()*tmp.norm()*cos<a, tmp>, as a and tmp are both normalized vector, 
    // a.transpose()*tmp = a.norm*tmp.norm*cos<a, tmp> = cos<a, tmp>
    // a * (a.transpose() * tmp) is the projection of tmp on the vector a
    // tmp - a * (a.transpose() * tmp) is a vector who is vertical to a in the plane determined by a and tmp
    b = (tmp - a * (a.transpose() * tmp)).normalized();
    c = a.cross(b);// c is a vector who is vertical to both vector a and vector b
    MatrixXd bc(3, 2);// the first column is vector b, the second column is vector c
    bc.block<3, 1>(0, 0) = b;
    bc.block<3, 1>(0, 1) = c;
    return bc;
}

/*
 * (1) position_bk_under_c0_coordinate = position_ck_under_c0_coordinate + shift_ck_to_bk_under_c0_coordinate
 *     = position_ck_under_c0_coordinate + R_bk_to_c0*t_ck_to_bk
 *     = s*vision_compute_position_ck_under_c0_coordinate + R_bk_to_c0*t_calib_c_to_b
 * use the same equations as the function LinearAlignment, but use g+lxly*dg as gravity, here g is a constant other than a variable 
 */
void RefineGravity(map<double, ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x)
{
    Vector3d g0 = g.normalized() * G.norm();// keep the norm and correct the direction
    Vector3d lx, ly;
    //VectorXd x;
    int all_frame_count = all_image_frame.size();
    int n_state = all_frame_count * 3 + 2 + 1;// the state vector contains [v_b(0), v_b(1), ... , v_b(nstate-1), weight_of_lx, weight_of_ly, s]

    MatrixXd A{n_state, n_state};
    A.setZero();
    VectorXd b{n_state};
    b.setZero();

    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    for(int k = 0; k < 4; k++)
    {
        MatrixXd lxly(3, 2);
        lxly = TangentBasis(g0);// compute two unit vectors who is vertical to each other and g0
        int i = 0;
        for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++)
        {
            frame_j = next(frame_i);

            MatrixXd tmp_A(6, 9);
            tmp_A.setZero();
            VectorXd tmp_b(6);
            tmp_b.setZero();

            double dt = frame_j->second.pre_integration->sum_dt;


            tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
            tmp_A.block<3, 2>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity() * lxly;
            tmp_A.block<3, 1>(0, 8) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0;     
            tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0] - frame_i->second.R.transpose() * dt * dt / 2 * g0;

            tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
            tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
            tmp_A.block<3, 2>(3, 6) = frame_i->second.R.transpose() * dt * Matrix3d::Identity() * lxly;
            tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v - frame_i->second.R.transpose() * dt * Matrix3d::Identity() * g0;


            Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
            //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
            //MatrixXd cov_inv = cov.inverse();
            cov_inv.setIdentity();

            MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
            VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

            A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
            b.segment<6>(i * 3) += r_b.head<6>();

            A.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
            b.tail<3>() += r_b.tail<3>();

            A.block<6, 3>(i * 3, n_state - 3) += r_A.topRightCorner<6, 3>();
            A.block<3, 6>(n_state - 3, i * 3) += r_A.bottomLeftCorner<3, 6>();
        }
        A = A * 1000.0;
        b = b * 1000.0;
        x = A.ldlt().solve(b);
        VectorXd dg = x.segment<2>(n_state - 3);// dg is a vector with two elements x(n_state-3) and x(n_state-2)
        g0 = (g0 + lxly * dg).normalized() * G.norm();//lxly's dimension is 3*2, dg's dimension is 2*1
        //double s = x(n_state - 1);
    }   
    g = g0;
}


/*
 * consider the IMU pre-integration incremental amout position:alpha_bk+1_to_bk and velocity:beta_bk+1_to_bk
 * delta_alpha_bk+1_to_bk = alpha_bk+1_to_bk - R_c0_to_bk*shift_from_bk+1_to_bk_under_c0_coordinate
 *  = alpha_bk+1_to_bk - R_c0_to_bk*(s*(position_bk+1_under_c0_coordinate-position_bk_under_c0_coordinate)
 *    -R_bk_to_c0*v_bk*delta_tk + 1/2*g_under_c0_coordinate*delta_tk*delta_tk)                           -------------------------------(1)
 * delta_beta_bk+1_to_bk = beta_bk+1_to_bk - R_c0_to_bk(R_bk+1_to_c0*v_bk+1 - R_bk_to_c0*v_bk +g_under_c0_coordinate*delta_tk)    ------(2)
 * consider the relationship between position_bk_under_c0_coordinate and position_ck_under_c0_coordinate, we have
 * position_bk_under_c0_coordinate = position_ck_under_c0_coordinate + shift_ck_to_bk_under_c0_coordinate
 *  = position_ck_under_c0_coordinate + R_bk_to_c0*position_ck_under_bk_coordinate
 *  = position_ck_under_c0_coordinate + R_bk_to_c0*t_calib_c_to_b                  -----------------------------------------------------(3)
 * use (3) to replace position_bk+1_under_c0_coordinate and position_bk_under_c0_coordinate in (1) and express (1)=0 and (2)=0 in matrix fomulation we get
 *                                                                                                                                      | v_bk                  |
 * | -I*delta_tk            0              1/2*R_c0_to_bk*delta_tk*delta_tk  R_c0_to_bk*(position_ck+1_under_c0-position_ck_under_c0) |*| v_bk+1                |
 * |     -I       R_c0_to_bk*R_bk+1_to_c0         Rc0_to_bk*delta_tk                                     0                            | | g_under_c0_coordinate |
 *                                                                                                                                      | s                     |
 * = | alpha_bk+1_to_bk - t_calib_c_to_b + R_c0_to_bk*R_bk+1_to_c0*t_calib_c_to_b |
 *   |                               beta_bk+1_to_bk                              |                            --------------------------(4)
 * the following equations are written according to (4)  
 */
// use the difference of the position-delta_alpha_bk+1_to_bk and velocity-delta_beta_bk+1_to_bk
bool LinearAlignment(map<double, ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x)
{
    int all_frame_count = all_image_frame.size();
    int n_state = all_frame_count * 3 + 3 + 1;// the state vector contains [v_b0, v_b1, ..., v_bn, g_c0, s], where s represents the scale
 
    MatrixXd A{n_state, n_state};
    A.setZero();
    VectorXd b{n_state};
    b.setZero();

    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    // for the class ImageFrame, I think the variable R and T represents the results of the visual computation
    int i = 0;
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++)
    {
        frame_j = next(frame_i);

        MatrixXd tmp_A(6, 10);
        tmp_A.setZero();
        VectorXd tmp_b(6);
        tmp_b.setZero();

        double dt = frame_j->second.pre_integration->sum_dt;

        tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
        // frame_i->second.R.transpose() refers to R_bk_to_c0
        tmp_A.block<3, 3>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity();
        // frame_j->second.T refers to position_of_ck+1_under_c0_coordinate
        // frame_i->second.T refers to position_of_ck_under_c0_coordinate
        tmp_A.block<3, 1>(0, 9) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0;
        // RIC is a vector of Eigen::Matrix3d, TIC is a vector of Eigen::Vector3d     
        // frame_j->second.pre_integration->delta_p refers to alpha_bk+1_to_bk
        // TIC[0] refers to t_calib_c_to_b
        tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0];
        //cout << "delta_p   " << frame_j->second.pre_integration->delta_p.transpose() << endl;
        tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
        tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
        tmp_A.block<3, 3>(3, 6) = frame_i->second.R.transpose() * dt * Matrix3d::Identity();
        // frame_j->second.pre_integration->delta_v refers to beta_bk+1_to_bk
        tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v;
        //cout << "delta_v   " << frame_j->second.pre_integration->delta_v.transpose() << endl;

        Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
        //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
        //MatrixXd cov_inv = cov.inverse();
        cov_inv.setIdentity();

        MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
        VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

        A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
        b.segment<6>(i * 3) += r_b.head<6>();

        A.bottomRightCorner<4, 4>() += r_A.bottomRightCorner<4, 4>();
        b.tail<4>() += r_b.tail<4>();

        // int n_state = all_image_frame.size() * 3 + 3 + 1;
        A.block<6, 4>(i * 3, n_state - 4) += r_A.topRightCorner<6, 4>();
        A.block<4, 6>(n_state - 4, i * 3) += r_A.bottomLeftCorner<4, 6>();
    }
    A = A * 1000.0;
    b = b * 1000.0;
    x = A.ldlt().solve(b);// the size of VectorXd x is n_state
    double s = x(n_state - 1) / 100.0;
    ROS_DEBUG("estimated scale: %f", s);
    g = x.segment<3>(n_state - 4);//get [x(n_state-4), x(n_state-3), x(n_state-2)]
    ROS_DEBUG_STREAM(" result g     " << g.norm() << " " << g.transpose());
    if(fabs(g.norm() - G.norm()) > 1.0 || s < 0) //Eigen::Vector3d G{0.0, 0.0, 9.8};
    {
        return false;
    }

    RefineGravity(all_image_frame, g, x);
    s = (x.tail<1>())(0) / 100.0;
    (x.tail<1>())(0) = s;
    ROS_DEBUG_STREAM(" refine     " << g.norm() << " " << g.transpose());
    if(s < 0.0 )
        return false;   
    else
        return true;
}

// align the visual and imu result, get bg, gravity and scale
bool VisualIMUAlignment(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs, Vector3d &g, VectorXd &x)
{
    // the objective function is that, the orientation between two frames should equals the pre-integration of imu
    // estimate a delta_bw who would minimum sum of all k two-norm(q_bk+1_to_c0.inverse*q_bk_to_c0*gamma_bk+1_to_bk)
    solveGyroscopeBias(all_image_frame, Bgs);

    // use the difference of the position-delta_alpha_bk+1_to_bk and velocity-delta_beta_bk+1_to_bk to compute the gravity and scale
    if(LinearAlignment(all_image_frame, g, x))
        return true;
    else 
        return false;
}
