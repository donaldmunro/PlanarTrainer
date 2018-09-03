#ifndef POSE_OTHER_0ab5f4ff_4891_4c59_bb65_14edf0af5d77_H
#define POSE_OTHER_0ab5f4ff_4891_4c59_bb65_14edf0af5d77_H

#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>

namespace poseother
{
   void essential_pose(const cv::Mat& intrinsics, const std::vector<cv::Point3d>& train_img_pts,
                       const std::vector<cv::Point3d>& query_img_pts);

   void FivePointRelativePose(const cv::Mat& intrinsics, const std::vector<cv::Point3d>& train_img_pts,
                              const std::vector<cv::Point3d>& query_img_pts,
                              std::vector<Eigen::Quaterniond>& Qs, std::vector<Eigen::Vector3d>& Ts);

   void TwoPointPosePartialRotation(const cv::Mat& intrinsics, const std::vector<cv::Point3d>& train_img_pts,
                                    const std::vector<cv::Point3d>& query_img_pts,
                                    const cv::Vec3f query_gravity,
                                    const cv::Vec3f down_axis, /*= Eigen::Vector3d(0, -1, 0) for Y down RH system ? */
                                    std::vector<Eigen::Quaterniond>& Qs, std::vector<Eigen::Vector3d>& Ts);

   double TwoPointPosePartialRotationRANSAC(const cv::Mat& intrinsics, const std::vector<cv::Point3d>& world_pts,
                                          const std::vector<cv::Point3d>& image_pts,
                                          const cv::Vec3f query_g, const cv::Vec3f down_axis,
                                          std::vector<Eigen::Quaterniond>& Qs, std::vector<Eigen::Vector3d>& Ts,
                                          void *RANSAC_params);
}

struct TheiaGravRansacModel
//======================
{
   TheiaGravRansacModel() {}

   TheiaGravRansacModel(Eigen::Quaterniond& rotation, Eigen::Vector3d& translation) :
         rotation(rotation), translation(translation)  { }

   TheiaGravRansacModel(const TheiaGravRansacModel& other) = default;

   TheiaGravRansacModel& operator=(const TheiaGravRansacModel &other) = default;

   Eigen::Quaterniond rotation;
   Eigen::Vector3d translation;
   double error = std::numeric_limits<double >::max();
};

#ifdef USE_THEIA_RANSAC
#include <tuple>

#include "sample_consensus_estimator.h"
#include "create_and_initialize_ransac_variant.h"

struct TheiaGravRansacEstimator : public theia::Estimator<std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d>,
                                                          TheiaGravRansacModel>
//=====================================================================================================================
{
   TheiaGravRansacEstimator(Eigen::Matrix3d& K, Eigen::Matrix3d& KI, Eigen::Quaterniond QI, Eigen::Vector3d& axis) :
         K(K), KI(KI), QI(QI), axis(axis) {}

   double SampleSize() const override { return 2; }

   bool EstimateModel(const std::vector<std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d>> &matches,
                      std::vector<TheiaGravRansacModel> *models) const override;

   double Error(const std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d>& match,
                const TheiaGravRansacModel& model) const override;

   Eigen::Matrix3d K, KI;
   Eigen::Quaterniond QI;
   Eigen::Vector3d axis;
};
#else
#include "Ransac.hh"

struct TheiaGravRansacData
//========================
{
   TheiaGravRansacData(const Eigen::Matrix3d& EKI, const Eigen::Quaterniond& rotateToAxis,
                       const std::vector<cv::Point3d> &world_pts, const std::vector<cv::Point3d> &query_img_pts)
   {
      for (size_t i=0; i<std::min(world_pts.size(), query_img_pts.size()); i++)
      {
         const cv::Point3d& tp  = world_pts[i];
         train_pts.emplace_back(tp.x, tp.y, tp.z);
         const cv::Point3d& qp  = query_img_pts[i];
         image_pts.emplace_back(qp.x, qp.y, qp.z);
         rotated_image_pts.emplace_back(rotateToAxis*(EKI*Eigen::Vector3d(qp.x, qp.y, qp.z)));
      }
   }

   std::vector<Eigen::Vector3d> train_pts;
   std::vector<Eigen::Vector3d> image_pts, rotated_image_pts;
   Eigen::Matrix3d OpenCV2OpenGL;
};

struct TheiaGravRansacEstimator
//=============================
{
   TheiaGravRansacEstimator(Eigen::Matrix3d& KI, const Eigen::Quaterniond Q, const Eigen::Quaterniond QI, Eigen::Vector3d& axis) :
         KI(KI), Q(Q), QI(QI), axis(axis) {}

   const int estimate(const TheiaGravRansacData& samples, const std::vector<size_t>& sampleIndices,
                      templransac::RANSACParams& parameters, std::vector<TheiaGravRansacModel>& models) const;

   const void error(const TheiaGravRansacData& samples, const std::vector<size_t>& sampleIndices,
                    TheiaGravRansacModel& model, std::vector<size_t>& inlier_indices,
                    std::vector<size_t>& outlier_indices, double error_threshold) const;

   Eigen::Matrix3d KI;
   Eigen::Quaterniond Q, QI;
   Eigen::Vector3d axis;
};
#endif

#endif //POSE_OTHER_0ab5f4ff_4891_4c59_bb65_14edf0af5d77_H
