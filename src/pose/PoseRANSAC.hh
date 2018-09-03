#ifndef _POSE2DRANSAC_H
#define _POSE2DRANSAC_H

#include <opencv2/core/core.hpp>

#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Eigen>

namespace pose2d
{
   void pose_translation(const Eigen::Matrix3d& Kinv, const std::vector<std::pair<cv::Point3d, cv::Point3d>>& pts,
                         const Eigen::Matrix3d& R, Eigen::Vector3d& translation);

   void pose_translation(const Eigen::Matrix3d& Kinv, const std::vector<cv::Point3d>& train_img_pts,
                         const std::vector<cv::Point3d>& query_img_pts,
                         const Eigen::Matrix3d& R, Eigen::Vector3d& translation);

   void pose_translation(const Eigen::Matrix3d& Kinv, const std::vector<std::pair<cv::Point3d, cv::Point3d>>& pts,
                         const double depth, const Eigen::Matrix3d& R, Eigen::Vector3d& translation);

   void pose_translation(const Eigen::Matrix3d& Kinv, const std::vector<cv::Point3d>& train_img_pts,
                         const std::vector<cv::Point3d>& query_img_pts, const double depth,
                         const Eigen::Matrix3d& R, Eigen::Vector3d& translation);
};

namespace pose3d
{
   void pose_translation(const std::vector<cv::Point3d>& world_pts, const std::vector<cv::Point2d>& query_image_pts,
                         const Eigen::Matrix3d& KI, const Eigen::Quaterniond& Q, Eigen::Vector3d& translation);

   void pose_translation(const std::vector<cv::Point3d>& world_pts, const std::vector<cv::Point2d>& query_image_pts,
                         const Eigen::Matrix3d& KI, const Eigen::Matrix3d& R, Eigen::Vector3d& translation);
};

struct GravPoseRansacModel
//=========================
{
   GravPoseRansacModel() : rotation(0, 0, 0, 0), translation(0, 0, 0) {}
   GravPoseRansacModel(const Eigen::Quaterniond& rotation_, const Eigen::Vector3d& translation_) : rotation(rotation_),
                                                                                                   translation(translation_) {}
   GravPoseRansacModel(const GravPoseRansacModel& other) = default;

   GravPoseRansacModel& operator=(const GravPoseRansacModel &other) = default;

   Eigen::Quaterniond rotation;
   Eigen::Vector3d translation;
};

#ifdef USE_THEIA_RANSAC
#include "sample_consensus_estimator.h"
#include "create_and_initialize_ransac_variant.h"

struct Grav2DRansacEstimator : public theia::Estimator<std::pair<cv::Point3d, cv::Point3d>, GravPoseRansacModel>
//=================================================================================================================
{
   Grav2DRansacEstimator(const cv::Mat& K_, const Eigen::Quaterniond& Q_, const double depth_ =0, int samples =3) :
      Q(Q_), R(Q_.toRotationMatrix()), depth(depth_), sample_size(samples)
   //--------------------------------------------------------------------------------------------
   {
      Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> EK_((double *) K_.data);
      EK = EK_;
      KI = EK.inverse();
   }

   Grav2DRansacEstimator(Eigen::Matrix3d _K, const Eigen::Quaterniond& Q_, const double depth_ =0, int samples =3) :
         EK(std::move(_K)), Q(Q_), R(Q_.toRotationMatrix()), depth(depth_), sample_size(samples) { KI = EK.inverse(); }

   virtual double SampleSize() const override { return sample_size; }

   bool EstimateModel(const std::vector<std::pair<cv::Point3d, cv::Point3d>> &matches,
                      std::vector<GravPoseRansacModel> *models) const override;

   double Error(const std::pair<cv::Point3d, cv::Point3d> &match, const GravPoseRansacModel &model) const override;

   Eigen::Matrix3d EK;
   const Eigen::Quaterniond Q;
   const Eigen::Matrix3d R;
   Eigen::Matrix3d KI;
   const double depth;
   int sample_size;
};

struct Grav3DRansacEstimator : public theia::Estimator<std::pair<cv::Point3d, cv::Point2d>, GravPoseRansacModel>
//=================================================================================================================
{
   Grav3DRansacEstimator(const Eigen::Matrix3d& KI_, const Eigen::Quaterniond& Q_, int samples =3) :
      KI(KI_), K(KI_.inverse()), Q(Q_), R(Q_.toRotationMatrix()), sample_size(samples) { }

   double SampleSize() const override { return sample_size; }

   bool EstimateModel(const std::vector<std::pair<cv::Point3d, cv::Point2d>> &matches,
                      std::vector<GravPoseRansacModel> *models) const override;

   double Error(const std::pair<cv::Point3d, cv::Point2d> &match, const GravPoseRansacModel &model) const override;

   Eigen::Matrix3d KI, K;
   const Eigen::Quaterniond Q;
   const Eigen::Matrix3d R;
   int sample_size;
};
#else
#include "Ransac.hh"

struct Grav2DRansacData
//=====================
{
   Grav2DRansacData(const std::vector<std::pair<cv::Point3d, cv::Point3d>>& pts) : pts(pts) {}

   const std::vector<std::pair<cv::Point3d, cv::Point3d>>& pts;
};

struct Grav3DRansacData
//=====================
{
   Grav3DRansacData(const std::vector<std::pair<cv::Point3d, cv::Point2d>>& pts) : pts(pts) {}

   const std::vector<std::pair<cv::Point3d, cv::Point2d>>& pts;
};

template <typename Pt>
static inline size_t RANSAC_copy_points(const Grav2DRansacData& samples, const std::vector<size_t>& sampleIndices,
                                        std::vector<Pt>& train_pts, std::vector<Pt>& query_pts)
//-----------------------------------------------------------------------------------------------------
{
   const size_t no = sampleIndices.size();
   for (size_t i=0; i<no; i++)
   {
      size_t index = sampleIndices[i];
      train_pts.emplace_back(samples.pts[index].first);
      query_pts.emplace_back(samples.pts[index].second);
   }
   return no;
}

static inline size_t copy_points3d(const Grav3DRansacData& samples, const std::vector<size_t>& sampleIndices,
                                   std::vector<cv::Point3d>& world_pts, std::vector<cv::Point2d>& query_pts)
//-----------------------------------------------------------------------------------------------------
{
   const size_t no = sampleIndices.size();
   for (size_t i=0; i<no; i++)
   {
      size_t index = sampleIndices[i];
      const cv::Point3d& wpt = samples.pts[index].first;
      const cv::Point2d& ipt = samples.pts[index].second;
      world_pts.emplace_back(wpt.x, wpt.y, wpt.z);
      query_pts.emplace_back(ipt.x, ipt.y);
   }
   return no;
}

struct PlanarRansacEstimator
//==========================
{
   PlanarRansacEstimator(const cv::Mat& K_, const Eigen::Quaterniond& Q_) : Q(Q_), R(Q_.toRotationMatrix())
   //-------------------------------------------------------------------------------------------
   {
      Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> EK_((double *) K_.data);
      EK = EK_;
      KI = EK.inverse();
   }

   PlanarRansacEstimator(Eigen::Matrix3d K_, const Eigen::Quaterniond& Q_) : EK(std::move(K_)), Q(Q_), R(Q_.toRotationMatrix())
   //-------------------------------------------------------------------------------------------
   {
      KI = EK.inverse();
   }

   Eigen::Matrix3d EK;
   const Eigen::Quaterniond Q;
   const Eigen::Matrix3d R;
   Eigen::Matrix3d KI;

};

struct Grav2DRansacEstimator : public PlanarRansacEstimator
//==============================================================
{
   Grav2DRansacEstimator(const cv::Mat& K_, const Eigen::Quaterniond& Q_) : PlanarRansacEstimator(K_, Q_) {}

   Grav2DRansacEstimator(const Eigen::Matrix3d K_, const Eigen::Quaterniond& Q_) : PlanarRansacEstimator(K_, Q_) {}

   const int estimate(const Grav2DRansacData &samples, const std::vector<size_t> &sampleIndices,
                      templransac::RANSACParams &parameters,
                      std::vector<GravPoseRansacModel> &models) const;

   const void error(const Grav2DRansacData &samples, const std::vector<size_t> &sampleIndices,
                    GravPoseRansacModel &model, std::vector<size_t> &inlier_indices,
                    std::vector<size_t> &outlier_indices, double error_threshold) const;
};

struct Grav2DDepthRansacEstimator : public PlanarRansacEstimator
//==============================================================
{
   Grav2DDepthRansacEstimator(const cv::Mat& K_, const Eigen::Quaterniond& Q_, const double depth_) :
         PlanarRansacEstimator(K_, Q_), depth(depth_) {}

   Grav2DDepthRansacEstimator(const Eigen::Matrix3d K_, const Eigen::Quaterniond& Q_, const double depth_) :
         PlanarRansacEstimator(K_, Q_), depth(depth_) {}

   const int estimate(const Grav2DRansacData& samples, const std::vector<size_t>& sampleIndices,
                      templransac::RANSACParams& parameters,
                      std::vector<GravPoseRansacModel>& models) const;

   const void error(const Grav2DRansacData& samples, const std::vector<size_t>& sampleIndices,
                    GravPoseRansacModel& model, std::vector<size_t>& inlier_indices,
                    std::vector<size_t>& outlier_indices, double error_threshold) const;

   const double depth;
};

struct Grav3DRansacEstimator
//==========================
{
   Grav3DRansacEstimator(const Eigen::Matrix<double, 3, 3>& KI_, const Eigen::Quaterniond& Q_)
         : Q(Q_), KI(KI_), K(KI.inverse()), R(Q_.toRotationMatrix()) { }

   const int estimate(const Grav3DRansacData& samples, const std::vector<size_t>& sampleIndices,
                      templransac::RANSACParams& parameters,
                      std::vector<GravPoseRansacModel>& models) const;

   const void error(const Grav3DRansacData& samples, const std::vector<size_t>& sampleIndices,
                    GravPoseRansacModel& model, std::vector<size_t>& inlier_indices,
                    std::vector<size_t>& outlier_indices, double error_threshold) const;

   const Eigen::Quaterniond Q;
   const Eigen::Matrix3d& KI;
   const Eigen::Matrix3d K, R;
};

#endif // USE_THEIA_RANSAC

#endif //_POSE2DRANSAC_H
