#pragma once

#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include "../utility/utility.h"

/*
 * sometimes the parameter x can overparameterize a problem. In that case it is desirable to choose a parameterization to remove
 * the null directions of the cost.More generally, if x lies on a manifold of a smaller dimension than the ambient space that it 
 * is embedded in, then it is numerically and computationally more effective to optimize it using a parameterization that lives
 * in the tangent space of tha manifold at each point.
 * GlobalSize returns the dimension of the ambient space in which the parameter block x lives
 * LocalSize returns the size of tangent space that delta_x lives in
 */
class PoseLocalParameterization : public ceres::LocalParameterization
{
    virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const;
    virtual bool ComputeJacobian(const double *x, double *jacobian) const;
    virtual int GlobalSize() const { return 7; };
    virtual int LocalSize() const { return 6; };
};
