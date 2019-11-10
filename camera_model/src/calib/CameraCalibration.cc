#include "camodocal/calib/CameraCalibration.h"

#include <cstdio>
#include <eigen3/Eigen/Dense>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/sparse_graph/Transform.h"
#include "camodocal/gpl/EigenQuaternionParameterization.h"
#include "camodocal/gpl/EigenUtils.h"
#include "camodocal/camera_models/CostFunctionFactory.h"

#include "ceres/ceres.h"
namespace camodocal
{

CameraCalibration::CameraCalibration()
 : m_boardSize(cv::Size(0,0))
 , m_squareSize(0.0f)
 , m_verbose(false)
{

}

CameraCalibration::CameraCalibration(const Camera::ModelType modelType,
                                     const std::string& cameraName,
                                     const cv::Size& imageSize,
                                     const cv::Size& boardSize,
                                     float squareSize)
 : m_boardSize(boardSize)
 , m_squareSize(squareSize)
 , m_verbose(false)
{
    m_camera = CameraFactory::instance()->generateCamera(modelType, cameraName, imageSize);
}

void
CameraCalibration::clear(void)
{
    m_imagePoints.clear();
    m_scenePoints.clear();
}

/*
 * push back std::vector<cv::Point2f> object corners to m_imagePoints, construct a std::vector<cv::Point3f> object scenePointsInView to store
 * the real 3D coordinate of the chessboard, then push back scenePointsInView to m_scenePoints
 */
void
CameraCalibration::addChessboardData(const std::vector<cv::Point2f>& corners)
{
    m_imagePoints.push_back(corners);

    std::vector<cv::Point3f> scenePointsInView;
    for (int i = 0; i < m_boardSize.height; ++i)
    {
        for (int j = 0; j < m_boardSize.width; ++j)
        {
            scenePointsInView.push_back(cv::Point3f(i * m_squareSize, j * m_squareSize, 0.0));
        }
    }
    m_scenePoints.push_back(scenePointsInView);
}

/*
 * brief:
 * compute intrinsic and r,t; do reprojection using the estimated params and compute reprojected error and its covariance
 * 
 * in detail:
 * 1. compute the intrinsic camera parameters and extrinsic parameters for each of the views
 * 2. use the parameters to project the 3D points to the image plane
 * 3. for every 2D measurement directly observed from the image in vector m_imagePoints, get the 2D coordinate directly from m_image, get
 *    the corresponding projected 2D points, use (measurement - projected) as the error
 * 4. compute the average error among all image points and their corresponding projected coordinate, use every element in the errVec and
 *    the average error to compute covariance
 * 5. return the output of calibrateHelper
 */
bool
CameraCalibration::calibrate(void)
{
    int imageCount = m_imagePoints.size();// how many images 

    // compute intrinsic camera parameters and extrinsic parameters for each of the views
    std::vector<cv::Mat> rvecs;
    std::vector<cv::Mat> tvecs;
    bool ret = calibrateHelper(m_camera, rvecs, tvecs);

    m_cameraPoses = cv::Mat(imageCount, 6, CV_64F);
    // store pose of every camera to m_cameraPoses
    for (int i = 0; i < imageCount; ++i)
    {
        m_cameraPoses.at<double>(i,0) = rvecs.at(i).at<double>(0);
        m_cameraPoses.at<double>(i,1) = rvecs.at(i).at<double>(1);
        m_cameraPoses.at<double>(i,2) = rvecs.at(i).at<double>(2);
        m_cameraPoses.at<double>(i,3) = tvecs.at(i).at<double>(0);
        m_cameraPoses.at<double>(i,4) = tvecs.at(i).at<double>(1);
        m_cameraPoses.at<double>(i,5) = tvecs.at(i).at<double>(2);
    }

    // Compute measurement covariance.
    std::vector<std::vector<cv::Point2f> > errVec(m_imagePoints.size());
    Eigen::Vector2d errSum = Eigen::Vector2d::Zero();
    size_t errCount = 0;
    for (size_t i = 0; i < m_imagePoints.size(); ++i)
    {
        std::vector<cv::Point2f> estImagePoints;
        // project the 3D real coordinate from the i th collection using pose in rvec and tvec, store the projected 2D coordinate to estImagePoints
        m_camera->projectPoints(m_scenePoints.at(i), rvecs.at(i), tvecs.at(i),
                                estImagePoints);

        /*
         * for every 2D measurement directly observed from the image in vector m_imagePoints, get the 2D coordinate directly from m_image, get the 
         * corresponding projected 2D coordinate, use (measurement-projected) as the error   
         */
        for (size_t j = 0; j < m_imagePoints.at(i).size(); ++j)
        {
            cv::Point2f pObs = m_imagePoints.at(i).at(j);
            cv::Point2f pEst = estImagePoints.at(j);

            cv::Point2f err = pObs - pEst;

            errVec.at(i).push_back(err);

            errSum += Eigen::Vector2d(err.x, err.y);
        }

        errCount += m_imagePoints.at(i).size();
    }

    // compute the average error among all image points and their corresponding projected coordinate
    Eigen::Vector2d errMean = errSum / static_cast<double>(errCount);

    Eigen::Matrix2d measurementCovariance = Eigen::Matrix2d::Zero();
    // use every element in the errVec and the average error to compute covariance
    for (size_t i = 0; i < errVec.size(); ++i)
    {
        for (size_t j = 0; j < errVec.at(i).size(); ++j)
        {
            cv::Point2f err = errVec.at(i).at(j);
            double d0 = err.x - errMean(0);
            double d1 = err.y - errMean(1);

            measurementCovariance(0,0) += d0 * d0;
            measurementCovariance(0,1) += d0 * d1;
            measurementCovariance(1,1) += d1 * d1;
        }
    }
    measurementCovariance /= static_cast<double>(errCount);
    measurementCovariance(1,0) = measurementCovariance(0,1);

    m_measurementCovariance = measurementCovariance;

    return ret;// return the output of calibrateHelper
}

// return how many views we get
int
CameraCalibration::sampleCount(void) const
{
    return m_imagePoints.size();
}

// return std::vector<std::vector<cv::Point2f> > object m_imagePoints
std::vector<std::vector<cv::Point2f> >&
CameraCalibration::imagePoints(void)
{
    return m_imagePoints;
}

// return const std::vector<std::vector<cv::point2f> > object m_imagePoints
const std::vector<std::vector<cv::Point2f> >&
CameraCalibration::imagePoints(void) const
{
    return m_imagePoints;
}

// return std::vector<std::vector<cv::Point3f> > object m_scenePoints
std::vector<std::vector<cv::Point3f> >&
CameraCalibration::scenePoints(void)
{
    return m_scenePoints;
}

// return const std::vector<std::vector<cv::Point3f> > object m_scenePoints
const std::vector<std::vector<cv::Point3f> >&
CameraCalibration::scenePoints(void) const
{
    return m_scenePoints;
}

// typedef boost::shared_ptr<Camera> CameraPtr
CameraPtr&
CameraCalibration::camera(void)
{
    return m_camera;
}

//typedef boost::shared_ptr<const Camera> CameraConstPtr
const CameraConstPtr
CameraCalibration::camera(void) const
{
    return m_camera;
}

Eigen::Matrix2d&
CameraCalibration::measurementCovariance(void)
{
    return m_measurementCovariance;
}

const Eigen::Matrix2d&
CameraCalibration::measurementCovariance(void) const
{
    return m_measurementCovariance;
}

cv::Mat&
CameraCalibration::cameraPoses(void)
{
    return m_cameraPoses;
}

const cv::Mat&
CameraCalibration::cameraPoses(void) const
{
    return m_cameraPoses;
}

/*
 * brief: 
 * draw circle around the 2D measurement and reprojection points on each image, compute average and max error for each image, and write
 * the error information on the image
 * 
 * in detail: 
 * 
 */
void
CameraCalibration::drawResults(std::vector<cv::Mat>& images) const
{
    std::vector<cv::Mat> rvecs, tvecs;

    // get rotation and translation of camera poses, push back separately into rvec and tvec 
    for (size_t i = 0; i < images.size(); ++i)
    {
        cv::Mat rvec(3, 1, CV_64F);
        rvec.at<double>(0) = m_cameraPoses.at<double>(i,0);
        rvec.at<double>(1) = m_cameraPoses.at<double>(i,1);
        rvec.at<double>(2) = m_cameraPoses.at<double>(i,2);

        cv::Mat tvec(3, 1, CV_64F);
        tvec.at<double>(0) = m_cameraPoses.at<double>(i,3);
        tvec.at<double>(1) = m_cameraPoses.at<double>(i,4);
        tvec.at<double>(2) = m_cameraPoses.at<double>(i,5);

        rvecs.push_back(rvec);
        tvecs.push_back(tvec);
    }

    int drawShiftBits = 4;
    int drawMultiplier = 1 << drawShiftBits;

    cv::Scalar green(0, 255, 0);
    cv::Scalar red(0, 0, 255);

    for (size_t i = 0; i < images.size(); ++i) // for every image in the image sequence
    {
        cv::Mat& image = images.at(i);
        if (image.channels() == 1)
        {
            cv::cvtColor(image, image, CV_GRAY2RGB);
        }

        // project the 3D points into 2D coordinate using r,t 
        std::vector<cv::Point2f> estImagePoints;
        m_camera->projectPoints(m_scenePoints.at(i), rvecs.at(i), tvecs.at(i),
                                estImagePoints);

        float errorSum = 0.0f;
        float errorMax = std::numeric_limits<float>::min();

        // for the i th image in the sequence, draw green circle around the measured 2D points, draw red circle around the estimated 2D 
        // points, compute norm between the 2D measurement and projected point as error, find the max error and average error for the points
        for (size_t j = 0; j < m_imagePoints.at(i).size(); ++j)
        {
            cv::Point2f pObs = m_imagePoints.at(i).at(j);
            cv::Point2f pEst = estImagePoints.at(j);

            cv::circle(image,
                       cv::Point(cvRound(pObs.x * drawMultiplier),
                                 cvRound(pObs.y * drawMultiplier)),
                       5, green, 2, CV_AA, drawShiftBits);

            // drawShiftBits means the shift number of fractional bits in the coordinates of the center and in the radius value
            cv::circle(image,
                       cv::Point(cvRound(pEst.x * drawMultiplier),
                                 cvRound(pEst.y * drawMultiplier)),
                       5, red, 2, CV_AA, drawShiftBits);

            float error = cv::norm(pObs - pEst);

            errorSum += error;
            if (error > errorMax)
            {
                errorMax = error;
            }
        }

        std::ostringstream oss;
        oss << "Reprojection error: avg = " << errorSum / m_imagePoints.at(i).size()
            << "   max = " << errorMax;

        cv::putText(image, oss.str(), cv::Point(10, image.rows - 10),
                    cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255, 255, 255),
                    1, CV_AA);
    }
}

void
CameraCalibration::writeParams(const std::string& filename) const
{
    m_camera->writeParametersToYamlFile(filename);
}

/*
 * write the following information to filename as binary file
 * 1. m_boardSize.width and m_boardSize.height
 * 2. m_squareSize
 * 3. m_measurementCovariance
 * 4. m_cameraPoses.rows, m_cameraPoses.cols, m_cameraPoses.type() and data in m_cameraPoses
 * 5. data in std::vector<std::vector<cv::Point2f> > m_imagePoints
 * 6. data in std::vector<std::vector<cv::Point3f> > m_scenePoints
 */
bool
CameraCalibration::writeChessboardData(const std::string& filename) const
{
    std::ofstream ofs(filename.c_str(), std::ios::out | std::ios::binary);
    if (!ofs.is_open())
    {
        return false;
    }

    writeData(ofs, m_boardSize.width);
    writeData(ofs, m_boardSize.height);
    writeData(ofs, m_squareSize);

    writeData(ofs, m_measurementCovariance(0,0));
    writeData(ofs, m_measurementCovariance(0,1));
    writeData(ofs, m_measurementCovariance(1,0));
    writeData(ofs, m_measurementCovariance(1,1));

    writeData(ofs, m_cameraPoses.rows);
    writeData(ofs, m_cameraPoses.cols);
    writeData(ofs, m_cameraPoses.type());
    for (int i = 0; i < m_cameraPoses.rows; ++i)
    {
        for (int j = 0; j < m_cameraPoses.cols; ++j)
        {
            writeData(ofs, m_cameraPoses.at<double>(i,j));
        }
    }

    writeData(ofs, m_imagePoints.size());
    for (size_t i = 0; i < m_imagePoints.size(); ++i)
    {
        writeData(ofs, m_imagePoints.at(i).size());
        for (size_t j = 0; j < m_imagePoints.at(i).size(); ++j)
        {
            const cv::Point2f& ipt = m_imagePoints.at(i).at(j);

            writeData(ofs, ipt.x);
            writeData(ofs, ipt.y);
        }
    }

    writeData(ofs, m_scenePoints.size());
    for (size_t i = 0; i < m_scenePoints.size(); ++i)
    {
        writeData(ofs, m_scenePoints.at(i).size());
        for (size_t j = 0; j < m_scenePoints.at(i).size(); ++j)
        {
            const cv::Point3f& spt = m_scenePoints.at(i).at(j);

            writeData(ofs, spt.x);
            writeData(ofs, spt.y);
            writeData(ofs, spt.z);
        }
    }

    return true;
}

bool
CameraCalibration::readChessboardData(const std::string& filename)
{
    std::ifstream ifs(filename.c_str(), std::ios::in | std::ios::binary);
    if (!ifs.is_open())
    {
        return false;
    }

    readData(ifs, m_boardSize.width);
    readData(ifs, m_boardSize.height);
    readData(ifs, m_squareSize);

    readData(ifs, m_measurementCovariance(0,0));
    readData(ifs, m_measurementCovariance(0,1));
    readData(ifs, m_measurementCovariance(1,0));
    readData(ifs, m_measurementCovariance(1,1));

    int rows, cols, type;
    readData(ifs, rows);
    readData(ifs, cols);
    readData(ifs, type);
    m_cameraPoses = cv::Mat(rows, cols, type);

    for (int i = 0; i < m_cameraPoses.rows; ++i)
    {
        for (int j = 0; j < m_cameraPoses.cols; ++j)
        {
            readData(ifs, m_cameraPoses.at<double>(i,j));
        }
    }

    size_t nImagePointSets;
    readData(ifs, nImagePointSets);

    m_imagePoints.clear();
    m_imagePoints.resize(nImagePointSets);
    for (size_t i = 0; i < m_imagePoints.size(); ++i)
    {
        size_t nImagePoints;
        readData(ifs, nImagePoints);
        m_imagePoints.at(i).resize(nImagePoints);

        for (size_t j = 0; j < m_imagePoints.at(i).size(); ++j)
        {
            cv::Point2f& ipt = m_imagePoints.at(i).at(j);
            readData(ifs, ipt.x);
            readData(ifs, ipt.y);
        }
    }

    size_t nScenePointSets;
    readData(ifs, nScenePointSets);

    m_scenePoints.clear();
    m_scenePoints.resize(nScenePointSets);
    for (size_t i = 0; i < m_scenePoints.size(); ++i)
    {
        size_t nScenePoints;
        readData(ifs, nScenePoints);
        m_scenePoints.at(i).resize(nScenePoints);

        for (size_t j = 0; j < m_scenePoints.at(i).size(); ++j)
        {
            cv::Point3f& spt = m_scenePoints.at(i).at(j);
            readData(ifs, spt.x);
            readData(ifs, spt.y);
            readData(ifs, spt.z);
        }
    }

    return true;
}

void
CameraCalibration::setVerbose(bool verbose)
{
    m_verbose = verbose;
}

/*
 * estimate intrinsics and extrinsics, then use ceres to do optimization
 */
bool
CameraCalibration::calibrateHelper(CameraPtr& camera,
                                   std::vector<cv::Mat>& rvecs, std::vector<cv::Mat>& tvecs) const
{
    rvecs.assign(m_scenePoints.size(), cv::Mat());// the size of rvecs is m_scenePoints.size(), each initialized to cv::Mat()
    tvecs.assign(m_scenePoints.size(), cv::Mat());

    // STEP 1: Estimate intrinsics
    camera->estimateIntrinsics(m_boardSize, m_scenePoints, m_imagePoints);

    // STEP 2: Estimate extrinsics        compute the pose for each frame 
    for (size_t i = 0; i < m_scenePoints.size(); ++i)
    {
        camera->estimateExtrinsics(m_scenePoints.at(i), m_imagePoints.at(i), rvecs.at(i), tvecs.at(i));
    }

    if (m_verbose)
    {
        std::cout << "[" << camera->cameraName() << "] "
                  << "# INFO: " << "Initial reprojection error: "
                  << std::fixed << std::setprecision(3)
                  << camera->reprojectionError(m_scenePoints, m_imagePoints, rvecs, tvecs)
                  << " pixels" << std::endl;
    }

    // STEP 3: optimization using ceres
    optimize(camera, rvecs, tvecs);

    if (m_verbose)
    {
        double err = camera->reprojectionError(m_scenePoints, m_imagePoints, rvecs, tvecs);
        std::cout << "[" << camera->cameraName() << "] " << "# INFO: Final reprojection error: "
                  << err << " pixels" << std::endl;
        std::cout << "[" << camera->cameraName() << "] " << "# INFO: "
                  << camera->parametersToString() << std::endl;
    }

    return true;
}

/*
 * brief:
 * use the intrinsics and extrinsics estimated before and the corresponding scene points and image points to construct the optimization problem, then
 * use the optimized params to reset intrinsics and extrinsics
 */
void
CameraCalibration::optimize(CameraPtr& camera,
                            std::vector<cv::Mat>& rvecs, std::vector<cv::Mat>& tvecs) const
{
    // Use ceres to do optimization
    ceres::Problem problem;

    // function rotationData() of the class Transform returns double*, rotation() of class Transform returns Eigen::Quaterniond& 
    // rvecs saves the rotation of frames, rvecs.size() means how many frames are there in the sequence
    std::vector<Transform, Eigen::aligned_allocator<Transform> > transformVec(rvecs.size());
    for (size_t i = 0; i < rvecs.size(); ++i)
    {
        Eigen::Vector3d rvec;
        cv::cv2eigen(rvecs.at(i), rvec);

        transformVec.at(i).rotation() = Eigen::AngleAxisd(rvec.norm(), rvec.normalized());
        transformVec.at(i).translation() << tvecs[i].at<double>(0),
                                            tvecs[i].at<double>(1),
                                            tvecs[i].at<double>(2);
    }

    std::vector<double> intrinsicCameraParams;
    // for different camera models, the intrinsic params are different
    m_camera->writeParameters(intrinsicCameraParams);

    // create residuals for each observation
    for (size_t i = 0; i < m_imagePoints.size(); ++i)// for each frame
    {
        for (size_t j = 0; j < m_imagePoints.at(i).size(); ++j)// for every image point in each frame
        {
            const cv::Point3f& spt = m_scenePoints.at(i).at(j);
            const cv::Point2f& ipt = m_imagePoints.at(i).at(j);

            // the number of the residuals is 2, the parameter blocks are camera intrinsic, quaternion and translation vector
            ceres::CostFunction* costFunction =
                CostFunctionFactory::instance()->generateCostFunction(camera,
                                                                      Eigen::Vector3d(spt.x, spt.y, spt.z),
                                                                      Eigen::Vector2d(ipt.x, ipt.y),
                                                                      CAMERA_INTRINSICS | CAMERA_POSE);

            ceres::LossFunction* lossFunction = new ceres::CauchyLoss(1.0);
            /*
             * AddResidualBlock() adds a residual block to the problem. It adds a CostFunction, an optional LossFunction and connects the 
             * CostFunction to a set of parameter block
             */
            problem.AddResidualBlock(costFunction, lossFunction,
                                     intrinsicCameraParams.data(),
                                     transformVec.at(i).rotationData(),
                                     transformVec.at(i).translationData());
        }

        ceres::LocalParameterization* quaternionParameterization =
            new EigenQuaternionParameterization;

        /*
         * void Problem::SetParameterization(double *values, LocalParamterization *local_parameterization) 
         * set the local parameterization for one of the parameter blocks.The local_parameterization is owned by the problem by default. It is 
         * acceptable to set the same parameterization for multiple parameters; the destructor is careful to delete local parameterization only 
         * once. The local parameterization can only be set once per parameter, and can not be changed once set.
         */
        // rotationData() returns double*
        problem.SetParameterization(transformVec.at(i).rotationData(),
                                    quaternionParameterization);
    }

    std::cout << "begin ceres" << std::endl;
    ceres::Solver::Options options;
    options.max_num_iterations = 1000;
    options.num_threads = 1;

    if (m_verbose)
    {
        options.minimizer_progress_to_stdout = true;
    }

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << "end ceres" << std::endl;

    if (m_verbose)
    {
        std::cout << summary.FullReport() << std::endl;
    }

    camera->readParameters(intrinsicCameraParams);// get and set intrinsics for the certain camera model

    // reset r and t vec for the 
    for (size_t i = 0; i < rvecs.size(); ++i)
    {
        Eigen::AngleAxisd aa(transformVec.at(i).rotation());

        Eigen::Vector3d rvec = aa.angle() * aa.axis();
        cv::eigen2cv(rvec, rvecs.at(i));

        cv::Mat& tvec = tvecs.at(i);
        tvec.at<double>(0) = transformVec.at(i).translation()(0);
        tvec.at<double>(1) = transformVec.at(i).translation()(1);
        tvec.at<double>(2) = transformVec.at(i).translation()(2);
    }
}

// read from in file stream to data
template<typename T>
void
CameraCalibration::readData(std::ifstream& ifs, T& data) const
{
    char* buffer = new char[sizeof(T)];

    ifs.read(buffer, sizeof(T));

    data = *(reinterpret_cast<T*>(buffer));

    delete[] buffer;
}

// write data to out file stream
template<typename T>
void
CameraCalibration::writeData(std::ofstream& ofs, T data) const
{
    char* pData = reinterpret_cast<char*>(&data);

    ofs.write(pData, sizeof(T));
}

}
