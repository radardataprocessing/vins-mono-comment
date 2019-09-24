#pragma once

#include <ros/assert.h>
#include <ceres/ceres.h>
#include <Eigen/Dense>
#include "../utility/utility.h"
#include "../utility/tic_toc.h"
#include "../parameters.h"

/*
 * in some cases, using automatic differentiation is not possible. For example, it may be the case that it is more efficient to compute the derivatives
 * in closed form instead of relying on the chain rule used by the automatic differentiation code. In such cases, it is possible to supply your own 
 * residuals and jacobian computation code. To do this, define a subclass of Cost function or SizedCostFunction if you know the sizes of the parameters
 * and residuals at compile time.
 * 2 means the residual has 2 dimension
 * the first 7 means the ith pose
 * the second 7 means the jth pose
 * the third 7 means the transform between imu and camera
 * 1 means the depth of the point
 */
class ProjectionFactor : public ceres::SizedCostFunction<2, 7, 7, 7, 1>
{
  public:
    ProjectionFactor(const Eigen::Vector3d &_pts_i, const Eigen::Vector3d &_pts_j);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;
    void check(double **parameters);

    Eigen::Vector3d pts_i, pts_j;
    Eigen::Matrix<double, 2, 3> tangent_base;
    static Eigen::Matrix2d sqrt_info;
    static double sum_t;
};
