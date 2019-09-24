#include "projection_factor.h"

Eigen::Matrix2d ProjectionFactor::sqrt_info; // the value of the variable sqrt_info is given in estimator.cpp
double ProjectionFactor::sum_t;

ProjectionFactor::ProjectionFactor(const Eigen::Vector3d &_pts_i, const Eigen::Vector3d &_pts_j) : pts_i(_pts_i), pts_j(_pts_j)
{
#ifdef UNIT_SPHERE_ERROR
    Eigen::Vector3d b1, b2;
    Eigen::Vector3d a = pts_j.normalized();
    Eigen::Vector3d tmp(0, 0, 1);
    if(a == tmp)
        tmp << 1, 0, 0;
    b1 = (tmp - a * (a.transpose() * tmp)).normalized();
    b2 = a.cross(b1);
    tangent_base.block<1, 3>(0, 0) = b1.transpose();
    tangent_base.block<1, 3>(1, 0) = b2.transpose();
#endif
};

/*
 * 
 */
bool ProjectionFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    TicToc tic_toc;
    // the ith pose
    Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    // the jth pose
    Eigen::Vector3d Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
    Eigen::Quaterniond Qj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

    // transform between imu and camera
    Eigen::Vector3d tic(parameters[2][0], parameters[2][1], parameters[2][2]);
    Eigen::Quaterniond qic(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);

    // depth of the point
    double inv_dep_i = parameters[3][0];

    // Eigen::Vector3d pts_i, pts_j
    Eigen::Vector3d pts_camera_i = pts_i / inv_dep_i; // compute the 3D position under ith camera coordinate
    Eigen::Vector3d pts_imu_i = qic * pts_camera_i + tic; // transform the 3D position from the ith camera coordinate to ith imu coordinate
    Eigen::Vector3d pts_w = Qi * pts_imu_i + Pi; // transform the 3D position from the ith imu coordinate to world coordinate
    Eigen::Vector3d pts_imu_j = Qj.inverse() * (pts_w - Pj); // transform the 3D position from the world coordinate to jth imu coordinate
    Eigen::Vector3d pts_camera_j = qic.inverse() * (pts_imu_j - tic);// transform the 3D position from the jth imu to jth camera coordinate
    Eigen::Map<Eigen::Vector2d> residual(residuals);
    /*
     * pts_camera_j = Ric.T() * (pts_imu_j - tic)
     * = Ric.T() * (Rj.T() * (pts_w - Pj) - tic)                                    ------------------------(1)
     * = Ric.T() * Rj.T() * pts_w - Ric.T() * Rj.T() * Pj - Ric.T()*tic             ------------------------(2)
     * = Ric.T() * Rj.T() * (Ri * pts_imu_i + Pi) - Ric.T() * Rj.T() * Pj - Ric.T()*tic                   ---------------(3)
     * = Ric.T() * Rj.T() * Ri * pts_imu_i + Ric.T() * Rj.T() * Pi - Ric.T() * Rj.T() * Pj - Ric.T()*tic            -------------------(4)
     * = Ric.T() * Rj.T() * Ri * (Ric * pts_camera_i + tic) + Ric.T() * Rj.T() * Pi - Ric.T() * Rj.T() * Pj - Ric.T()*tic       -------------------(5)
     * = Ric.T() * Rj.T() * Ri * Ric * pts_camera_i + Ric.T() * Rj.T() * Ri * tic + Ric.T() * Rj.T() * Pi - Ric.T() * Rj.T() * Pj - Ric.T()*tic  -----(6)
     * = Ric.T() * Rj.T() * Ri * Ric * (pts_i / inv_dep_i) + Ric.T() * Rj.T() * Ri * tic
     *   + Ric.T() * Rj.T() * Pi - Ric.T() * Rj.T() * Pj - Ric.T()*tic                        --------------------------------------------------------(7)
     */

#ifdef UNIT_SPHERE_ERROR 
    // the dimension of tangent_base is 2*3, (2*3)*(3*1) is 2*1, which is the dimension of the residual 
    residual =  tangent_base * (pts_camera_j.normalized() - pts_j.normalized());
#else
    double dep_j = pts_camera_j.z();// take the z value of the pts_camera_j as the depth of the point under jth camera coordinate
    // .head<n> means to get the first n values in the vector
    residual = (pts_camera_j / dep_j).head<2>() - pts_j.head<2>();// 
#endif

    // Eigen::Matrix2d ProjectionFactor::sqrt_info;
    residual = sqrt_info * residual;

    if (jacobians) // std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians;
    {
        Eigen::Matrix3d Ri = Qi.toRotationMatrix();
        Eigen::Matrix3d Rj = Qj.toRotationMatrix();
        Eigen::Matrix3d ric = qic.toRotationMatrix();
        Eigen::Matrix<double, 2, 3> reduce(2, 3);
#ifdef UNIT_SPHERE_ERROR
        double norm = pts_camera_j.norm();
        Eigen::Matrix3d norm_jaco;
        double x1, x2, x3;
        x1 = pts_camera_j(0);
        x2 = pts_camera_j(1);
        x3 = pts_camera_j(2);
        norm_jaco << 1.0 / norm - x1 * x1 / pow(norm, 3), - x1 * x2 / pow(norm, 3),            - x1 * x3 / pow(norm, 3),
                     - x1 * x2 / pow(norm, 3),            1.0 / norm - x2 * x2 / pow(norm, 3), - x2 * x3 / pow(norm, 3),
                     - x1 * x3 / pow(norm, 3),            - x2 * x3 / pow(norm, 3),            1.0 / norm - x3 * x3 / pow(norm, 3);
        reduce = tangent_base * norm_jaco;
#else
        // | 1/z  0  -x/(z*z) |
        // |  0  1/z -y/(z*z) | this is the jacobian of | x/z-x_measure  y/z-y_measure |'s derivative about | x y z |
        reduce << 1. / dep_j, 0, -pts_camera_j(0) / (dep_j * dep_j),
            0, 1. / dep_j, -pts_camera_j(1) / (dep_j * dep_j);
#endif
        reduce = sqrt_info * reduce;

        /*
         * cross-product axb=[a]x*b=-bxa=-[b]x*a
         * use ~ to represent is approximately equal to
         * for a rotation matrix's derivative about rotation:
         * a(R*p)/a(theta = lim(delta_theta->0) ((exp([delta_theta]x)*exp([theta]x)*p-exp([theta]x)*p)/delta_theta)
         * use Taylor expansion, we can get
         * ~ lim(delta_theta->0) (((I+[delta_theta]x)*exp([theta]x)*p-exp([theta]x)*p)/delta_theta)
         * = lim(delta_theta->0) ([delta_theta]x*exp([theta]x)*p/delta_theta)
         * = lim(delta_theta->0) ([delta_theta]x*R*p/delta_theta)
         * = lim(delta_theta->0) (-[R*p]x*delta_theta/delta_theta)
         * = -[R*p]x
         */
        if (jacobians[0])
        {
            // pts_camera_j = Ric.T() * Rj.T() * Ri * pts_imu_i + Ric.T() * Rj.T() * Pi - Ric.T() * Rj.T() * Pj - Ric.T()*tic
            //(2*3)*(3*6) is 2*6 and the 7th col is 0
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);

            // compute the derivative of the above comment function (4)
            Eigen::Matrix<double, 3, 6> jaco_i;
            jaco_i.leftCols<3>() = ric.transpose() * Rj.transpose();
            jaco_i.rightCols<3>() = ric.transpose() * Rj.transpose() * Ri * -Utility::skewSymmetric(pts_imu_i);

            jacobian_pose_i.leftCols<6>() = reduce * jaco_i;
            jacobian_pose_i.rightCols<1>().setZero();
        }

        if (jacobians[1])
        {
            // pts_camera_j = Ric.T() * (Rj.T() * (pts_w - Pj) - tic)
            // compute the derivative of the above comment function (1)
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[1]);

            Eigen::Matrix<double, 3, 6> jaco_j;
            jaco_j.leftCols<3>() = ric.transpose() * -Rj.transpose();
            jaco_j.rightCols<3>() = ric.transpose() * Utility::skewSymmetric(pts_imu_j);

            jacobian_pose_j.leftCols<6>() = reduce * jaco_j;
            jacobian_pose_j.rightCols<1>().setZero();
        }
        if (jacobians[2])
        {
            // pts_camera_j = Ric.T() * Rj.T() * Ri * Ric * pts_camera_i + Ric.T() * Rj.T() * Ri * tic
            //                + Ric.T() * Rj.T() * Pi - Ric.T() * Rj.T() * Pj - Ric.T()*tic
            // compute the derivative of the above comment function (6)
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_ex_pose(jacobians[2]);
            Eigen::Matrix<double, 3, 6> jaco_ex;
            jaco_ex.leftCols<3>() = ric.transpose() * (Rj.transpose() * Ri - Eigen::Matrix3d::Identity());
            Eigen::Matrix3d tmp_r = ric.transpose() * Rj.transpose() * Ri * ric;
            jaco_ex.rightCols<3>() = -tmp_r * Utility::skewSymmetric(pts_camera_i) + Utility::skewSymmetric(tmp_r * pts_camera_i) +
                                     Utility::skewSymmetric(ric.transpose() * (Rj.transpose() * (Ri * tic + Pi - Pj) - tic));
            jacobian_ex_pose.leftCols<6>() = reduce * jaco_ex;
            jacobian_ex_pose.rightCols<1>().setZero();
        }
        if (jacobians[3])
        {
            // pts_camera_j = Ric.T() * Rj.T() * Ri * Ric * (pts_i / inv_dep_i) + Ric.T() * Rj.T() * Ri * tic       
            //                + Ric.T() * Rj.T() * Pi - Ric.T() * Rj.T() * Pj - Ric.T()*tic          
            // compute the jacobian of the above comment function (7)
            Eigen::Map<Eigen::Vector2d> jacobian_feature(jacobians[3]);
#if 1
            jacobian_feature = reduce * ric.transpose() * Rj.transpose() * Ri * ric * pts_i * -1.0 / (inv_dep_i * inv_dep_i);
#else
            jacobian_feature = reduce * ric.transpose() * Rj.transpose() * Ri * ric * pts_i;
#endif
        }
    }
    sum_t += tic_toc.toc();

    return true;
}

/*
 * 
 */
void ProjectionFactor::check(double **parameters)
{
    double *res = new double[15];
    double **jaco = new double *[4];
    jaco[0] = new double[2 * 7];
    jaco[1] = new double[2 * 7];
    jaco[2] = new double[2 * 7];
    jaco[3] = new double[2 * 1];
    Evaluate(parameters, res, jaco);
    /*
     * puts : writes the C string pointed by str to the standard output (stdout) and appends a newline character('\n')
     * the function begins copying from the address specified (str) until it reaches the terminating null character('\0'). This 
     * terminating null-character is not copied to the stream
     * Notice that puts not only differs from fputs in that it uses stdout as destination, but it also appends a newline
     * character at the end automatically(which fputs does not)
     */
    puts("check begins");

    puts("my");

    std::cout << Eigen::Map<Eigen::Matrix<double, 2, 1>>(res).transpose() << std::endl
              << std::endl;
    std::cout << Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>>(jaco[0]) << std::endl
              << std::endl;
    std::cout << Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>>(jaco[1]) << std::endl
              << std::endl;
    std::cout << Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>>(jaco[2]) << std::endl
              << std::endl;
    std::cout << Eigen::Map<Eigen::Vector2d>(jaco[3]) << std::endl
              << std::endl;

    Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    Eigen::Vector3d Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
    Eigen::Quaterniond Qj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

    Eigen::Vector3d tic(parameters[2][0], parameters[2][1], parameters[2][2]);
    Eigen::Quaterniond qic(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);
    double inv_dep_i = parameters[3][0];

    Eigen::Vector3d pts_camera_i = pts_i / inv_dep_i;
    Eigen::Vector3d pts_imu_i = qic * pts_camera_i + tic;
    Eigen::Vector3d pts_w = Qi * pts_imu_i + Pi;
    Eigen::Vector3d pts_imu_j = Qj.inverse() * (pts_w - Pj);
    Eigen::Vector3d pts_camera_j = qic.inverse() * (pts_imu_j - tic);


    Eigen::Vector2d residual;
#ifdef UNIT_SPHERE_ERROR 
    residual =  tangent_base * (pts_camera_j.normalized() - pts_j.normalized());
#else
    double dep_j = pts_camera_j.z();
    residual = (pts_camera_j / dep_j).head<2>() - pts_j.head<2>();// the residual is the difference of the predicted and measured pixel coordinate in jth camera frame
#endif
    residual = sqrt_info * residual;

    puts("num");
    std::cout << residual.transpose() << std::endl;

    const double eps = 1e-6;
    Eigen::Matrix<double, 2, 19> num_jacobian;
    /*
     * k = 0 corresponds to a=0, b=0, pi=pi+(eps, 0, 0)
     * k = 1 corresponds to a=0, b=1, pi=pi+(0, eps, 0)
     * k = 2 corresponds to a=0, b=2, pi=pi+(0, 0, eps)
     * k = 3 corresponds to a=1, b=0, Qi=Qi*(1, eps/2, 0, 0)
     * k = 4 corresponds to a=1, b=1, Qi=Qi*(1, 0, eps/2, 0)
     * k = 5 corresponds to a=1, b=2, Qi=Qi*(1, 0, 0, eps/2)
     * k = 6 corresponds to a=2, b=0, pj=pj+(eps, 0, 0)
     * k = 7 corresponds to a=2, b=1, pj=pj+(0, eps, 0)
     * k = 8 corresponds to a=2, b=2, pj=pj+(0, 0, eps)
     * k = 9 corresponds to a=3, b=0, Qj=Qj*(1, eps/2, 0, 0)
     * k = 10 corresponds to a=3, b=1, Qj=Qj*(1, 0, eps/2, 0)
     * k = 11 corresponds to a=3, b=2, Qj=Qj*(1, 0, 0, eps/2)
     * k = 12 corresponds to a=4, b=0, tic=tic+(eps, 0, 0)
     * k = 13 corresponds to a=4, b=1, tic=tic+(0, eps, 0)
     * k = 14 corresponds to a=4, b=2, tic=tic+(0, 0, eps)
     * k = 15 corresponds to a=5, b=0, ric=ric*(1, eps/2, 0, 0)
     * k = 16 corresponds to a=5, b=1, ric=ric*(1, 0, eps/2, 0)
     * k = 17 corresponds to a=5, b=2, ric=ric*(1, 0, 0, eps/2)
     * k = 18 corresponds to a=6, b=0, inv_dep_i=inv_dep_i+eps
     */
    for (int k = 0; k < 19; k++)
    {
        Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

        Eigen::Vector3d Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
        Eigen::Quaterniond Qj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

        Eigen::Vector3d tic(parameters[2][0], parameters[2][1], parameters[2][2]);
        Eigen::Quaterniond qic(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);
        double inv_dep_i = parameters[3][0];

        int a = k / 3, b = k % 3;
        // if k%3==0, delta=(1, 0, 0)*eps; if k%3==1, delta=(0, 1, 0)*eps; if k%3==2, delta=(0, 0, 1)*eps
        Eigen::Vector3d delta = Eigen::Vector3d(b == 0, b == 1, b == 2) * eps;

        if (a == 0)
            Pi += delta;
        else if (a == 1)
            // deltaQ returns a quaternion dq whose w is 1, x is theta.x/2, y is theta.y/2, z is theta.z/2
            Qi = Qi * Utility::deltaQ(delta);
        else if (a == 2)
            Pj += delta;
        else if (a == 3)
            Qj = Qj * Utility::deltaQ(delta);
        else if (a == 4)
            tic += delta;
        else if (a == 5)
            qic = qic * Utility::deltaQ(delta);
        else if (a == 6)
            inv_dep_i += delta.x();

        Eigen::Vector3d pts_camera_i = pts_i / inv_dep_i;
        Eigen::Vector3d pts_imu_i = qic * pts_camera_i + tic;
        Eigen::Vector3d pts_w = Qi * pts_imu_i + Pi;
        Eigen::Vector3d pts_imu_j = Qj.inverse() * (pts_w - Pj);
        Eigen::Vector3d pts_camera_j = qic.inverse() * (pts_imu_j - tic);
        // for every k, use the new parameters to project the point into jth camera frame, compute the new residual and compare it with the original residual

        Eigen::Vector2d tmp_residual;
#ifdef UNIT_SPHERE_ERROR 
        tmp_residual =  tangent_base * (pts_camera_j.normalized() - pts_j.normalized());
#else
        double dep_j = pts_camera_j.z();
        tmp_residual = (pts_camera_j / dep_j).head<2>() - pts_j.head<2>();
#endif
        tmp_residual = sqrt_info * tmp_residual;
        num_jacobian.col(k) = (tmp_residual - residual) / eps;
    }
    std::cout << num_jacobian << std::endl;
}
