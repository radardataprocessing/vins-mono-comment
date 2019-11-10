#include "utility.h"

/*
 * compute the rotation R to transform the normalized gravity vector to (0, 0, 1), compute ypr(yaw, pitch, roll) using the R matrix, the yaw is the last axis to rotate around, and
 * the destina is parallal to the z axis, so if we rotate around z axis last, it doesn't affect the result, so return r=ypr2R(-yaw, 0, 0)*R
 */
Eigen::Matrix3d Utility::g2R(const Eigen::Vector3d &g)
{
    Eigen::Matrix3d R0;
    Eigen::Vector3d ng1 = g.normalized();
    Eigen::Vector3d ng2{0, 0, 1.0};
    R0 = Eigen::Quaterniond::FromTwoVectors(ng1, ng2).toRotationMatrix();
    double yaw = Utility::R2ypr(R0).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    // R0 = Utility::ypr2R(Eigen::Vector3d{-90, 0, 0}) * R0;
    return R0;
}
