#include <camodocal/sparse_graph/Transform.h>

namespace camodocal
{

// return an object of this class whose m_q is identity and m_t is zero
Transform::Transform()
{
    m_q.setIdentity();
    m_t.setZero();
}

// return an object of this class whose m_q is the quaternion expression of the top left 3*3 block of input H 
// and m_t is the top right 3*1 block of the input H
Transform::Transform(const Eigen::Matrix4d& H)
{
   m_q = Eigen::Quaterniond(H.block<3,3>(0,0));
   m_t = H.block<3,1>(0,3);
}

Eigen::Quaterniond&
Transform::rotation(void)
{
    return m_q;
}

const Eigen::Quaterniond&
Transform::rotation(void) const
{
    return m_q;
}

double*
Transform::rotationData(void)
{
    return m_q.coeffs().data();
}

const double* const
Transform::rotationData(void) const
{
    return m_q.coeffs().data();
}

Eigen::Vector3d&
Transform::translation(void)
{
    return m_t;
}

const Eigen::Vector3d&
Transform::translation(void) const
{
    return m_t;
}

double*
Transform::translationData(void)
{
    return m_t.data();
}

const double* const
Transform::translationData(void) const
{
    return m_t.data();
}

// put the rotation matrix expression of m_q to the left top 3*3 block of the output matrix and
// put the translation m_t to the top right 3*1 block of the output matrix
Eigen::Matrix4d
Transform::toMatrix(void) const
{
    Eigen::Matrix4d H;
    H.setIdentity();
    H.block<3,3>(0,0) = m_q.toRotationMatrix();
    H.block<3,1>(0,3) = m_t;

    return H;
}

}
