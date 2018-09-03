#include <iostream>
#include <cmath>
#include <chrono>
#include <limits>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/core/eigen.hpp>

#include "pose2d.h"
#include "Ransac.hh"
#include "PoseRANSAC.hh"

//#define USE_QUATERNION
#define USE_ROTATION_MATRIX

namespace pose2d
{
/* Matrix coefficients for the non-depth case matrix to be solved by SVD */
   inline double tx_coeff(double xp1, double yp1, double xq1, double yq1, const Eigen::Matrix3d& R)
//---------------------------------------------------------------------------------------------------
   {
      return (R(1, 2) + R(1, 0) * xp1 + R(1, 1) * yp1 - R(2, 2) * yq1 + (-(R(2, 0) * xp1) - R(2, 1) * yp1) * yq1);
      //return (R(1,2) + R(1,0)*xq1 - R(2,2)*yp1 - R(2,0)*xq1*yp1 + R(1,1)*yq1 - R(2,1)*yp1*yq1);
   }

   inline double ty_coeff(double xp1, double yp1, double xq1, double yq1, const Eigen::Matrix3d& R)
//----------------------------------------------------------------------------------------------------
   {
      return (-R(0, 2) - R(0, 0) * xp1 + R(2, 2) * xq1 + R(2, 0) * xp1 * xq1 - R(0, 1) * yp1 + R(2, 1) * xq1 * yp1);
      //return ((-R(0,2) + R(2,2)*xp1 - R(0,0)*xq1 + R(2,0)*xp1*xq1 - R(0,1)*yq1 + R(2,1)*xp1*yq1));
   }

   inline double tz_coeff(double xp1, double yp1, double xq1, double yq1, const Eigen::Matrix3d& R)
//---------------------------------------------------------------------------------------------------
   {
      return (-(R(1, 2) * xq1) - R(1, 0) * xp1 * xq1 - R(1, 1) * xq1 * yp1 +
              (R(0, 2) + R(0, 0) * xp1 + R(0, 1) * yp1) * yq1);
//   return (-(R(1,2)*xp1) - R(1,0)*xp1*xq1 + R(0,2)*yp1 + R(0,0)*xq1*yp1 - R(1,1)*xp1*yq1 + R(0,1)*yp1*yq1);
   }

/* RHS values for depth solution*/
   inline double b0(const Eigen::Matrix3d& R, const double x_0, const double y_0, const double d_0, const double y_1)
   {
      return d_0 *
             (-R(1, 0) * x_0 - R(1, 1) * y_0 - R(1, 2) + R(2, 0) * x_0 * y_1 + R(2, 1) * y_0 * y_1 + R(2, 2) * y_1);
   }

   inline double b1(const Eigen::Matrix3d& R, const double x_0, const double y_0, const double d_0, const double x_1)
   {
      return -d_0 *
             (-R(0, 0) * x_0 - R(0, 1) * y_0 - R(0, 2) + R(2, 0) * x_0 * x_1 + R(2, 1) * x_1 * y_0 + R(2, 2) * x_1);
   }

   inline double b2(const Eigen::Matrix3d& R, const double x_0, const double y_0, const double d_0,
                    const double x_1, const double y_1)
   {
      return d_0 *
             (-R(0, 0) * x_0 * y_1 - R(0, 1) * y_0 * y_1 - R(0, 2) * y_1 + R(1, 0) * x_0 * x_1 + R(1, 1) * x_1 * y_0 +
              R(1, 2) * x_1);
   }

   inline double b0(double Qw, double Qx, double Qy, double Qz, double x_0, double y_0, double y_1, double d)
   {
      double Qw2 = Qw * Qw, Qx2 = Qx * Qx, Qy2 = Qy * Qy, Qz2 = Qz * Qz, QwQx = Qw * Qx;
      return d * (-Qw2 * y_1 - 2 * QwQx * y_0 * y_1 - 2 * QwQx + 2 * Qw * Qy * x_0 * y_1 + Qx2 * y_1 -
                  2 * Qx * Qz * x_0 * y_1 +
                  Qy2 * y_1 - 2 * Qy * Qz * y_0 * y_1 + 2 * Qy * Qz - Qz2 * y_1 + 2 * x_0 * (Qw * Qz + Qx * Qy) +
                  y_0 * (Qw2 -
                         Qx2 + Qy2 - Qz2));
   }

   inline double b1(double Qw, double Qx, double Qy, double Qz, double x_0, double y_0, double x_1, double d)
   {
      double Qw2 = Qw * Qw, Qx2 = Qx * Qx, Qy2 = Qy * Qy, Qz2 = Qz * Qz, QwQy = Qw * Qy, QxQz = Qx * Qz;
      return -d *
             (-Qw2 * x_1 - 2 * Qw * Qx * x_1 * y_0 + 2 * QwQy * x_0 * x_1 + 2 * QwQy - 2 * Qw * Qz * y_0 + Qx2 * x_1 +
              2 * Qx * Qy * y_0 - 2 * QxQz * x_0 * x_1 + 2 * QxQz + Qy2 * x_1 - 2 * Qy * Qz * x_1 * y_0 - Qz2 * x_1 +
              x_0 * (Qw2 + Qx2 - Qy2 - Qz2));
   }

   inline double b2(double Qw, double Qx, double Qy, double Qz, double x_0, double y_0, double x_1, double y_1, double d)
   {
      double Qw2 = Qw * Qw, Qx2 = Qx * Qx, Qy2 = Qy * Qy, Qz2 = Qz * Qz, QwQz = Qw * Qz, QxQy = Qx * Qy;
      return -d *
             (-Qw2 * x_0 * y_1 - 2 * Qw * Qx * x_1 - 2 * Qw * Qy * y_1 + 2 * QwQz * x_0 * x_1 + 2 * QwQz * y_0 * y_1 -
              Qx2 * x_0 * y_1 + 2 * QxQy * x_0 * x_1 - 2 * QxQy * y_0 * y_1 - 2 * Qx * Qz * y_1 + Qy2 * x_0 * y_1 +
              2 * Qy * Qz * x_1 + Qz2 * x_0 * y_1 + x_1 * y_0 * (Qw2 - Qx2 + Qy2 - Qz2));

   }

   inline Eigen::Vector3d homogeneous(Eigen::Matrix3d A)
   //---------------------------------------------------------------------------------------------------------------------------
   {
      Eigen::JacobiSVD<Eigen::MatrixXd, Eigen::FullPivHouseholderQRPreconditioner>
            svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
      auto U = svd.matrixU();
      auto V = svd.matrixV();
      return V.col(V.cols() - 1);
   }

   inline Eigen::Quaterniond rotation(const Eigen::Vector3d &from, const Eigen::Vector3d &to,
                                      const Eigen::Vector3d &fallbackAxis = Eigen::Vector3d(0, 0, 0))
   //-----------------------------------------------------------------------------------------------
   {
      Eigen::Quaterniond q;
      Eigen::Vector3d v0 = from;
      Eigen::Vector3d v1 = to;
      v0.normalize();
      v1.normalize();

      double d = v0.dot(v1);
      if (d >= 1.0f)
         return Eigen::Quaterniond(1, 0, 0, 0);

      if (d < (1e-6f - 1.0f))
      {
         if (fallbackAxis != Eigen::Vector3d(0, 0, 0))
            q = Eigen::AngleAxis<double>(PI, fallbackAxis);
         else
         {
            // Generate an axis
            Eigen::Vector3d axis = Eigen::Vector3d(1, 0, 0).cross(from);
            if (axis.norm() < 0.000000001) // pick another if colinear
               axis = Eigen::Vector3d(0, 1, 0).cross(from);
            axis.normalize();
            q = Eigen::AngleAxis<double>(PI, axis);
         }
      }
      else
      {
         double s = sqrt((1 + d) * 2);
         double invs = 1 / s;

         Eigen::Vector3d c = v0.cross(v1);

         q.x() = c.x() * invs;
         q.y() = c.y() * invs;
         q.z() = c.z() * invs;
         q.w() = s * 0.5f;
         q.normalize();
      }
      return q;
   }

   void pose(const std::vector<std::pair<cv::Point3d, cv::Point3d>>& pts, const Eigen::Vector3d& train_g,
             const Eigen::Vector3d query_g, const cv::Mat& intrinsics,
             Eigen::Quaterniond& Q, Eigen::Vector3d& translation)
//-----------------------------------------------------------------------------------------------------------
   {
      Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> K((double*) intrinsics.data);
      pose(pts, train_g, query_g, K, Q, translation);
   }

   void pose(const std::vector<std::pair<cv::Point3d, cv::Point3d>>& pts,
             const Eigen::Vector3d& train_g, const Eigen::Vector3d query_g,
             const Eigen::Matrix3d& K, Eigen::Quaterniond& Q, Eigen::Vector3d& translation)
   //--------------------------------------------------------------------------------------
   {
      Eigen::Matrix3d KI = K.inverse();
      //   std::cout << K << std::endl << KI << std::endl;
      //   if (std::isnan(KI(0,0))) KI = K;

      Eigen::Vector3d train_gravity(train_g), query_gravity(query_g);
      train_gravity.normalize();
      query_gravity.normalize();
      //   Q = rotation(train_gravity, query_gravity); // camera to model
      Q = rotation(query_gravity, train_gravity); // model to camera
      Eigen::Matrix3d R = Q.toRotationMatrix();
      size_t m = pts.size();
      Eigen::MatrixXd A(m, 3);
      for (size_t row = 0; row < m; row++)
      {
         const cv::Point3d& tpt = pts[row].first;
         const cv::Point3d& qpt = pts[row].second;
         Eigen::Vector3d train_ray = KI*Eigen::Vector3d(tpt.x, tpt.y, 1);
         Eigen::Vector3d query_ray = KI*Eigen::Vector3d(qpt.x, qpt.y, 1);
         double xt1 = train_ray[0], yt1 = train_ray[1], xq1 = query_ray[0], yq1 = query_ray[1];
         A.row(row) << tx_coeff(xt1, yt1, xq1, yq1, R),
               ty_coeff(xt1, yt1, xq1, yq1, R),
               tz_coeff(xt1, yt1, xq1, yq1, R);
      }

      Eigen::JacobiSVD<Eigen::MatrixXd, Eigen::FullPivHouseholderQRPreconditioner>
            svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
      auto V = svd.matrixV();
      translation = V.col(V.cols() - 1);
//   assert(mut::check_essential(R, translation));
   }

   void pose(const std::vector<std::pair<cv::Point3d, cv::Point3d>>& pts,
             const Eigen::Vector3d& train_g, const Eigen::Vector3d query_g, const double depth,
             const cv::Mat& intrinsics, Eigen::Quaterniond& Q, Eigen::Vector3d& translation)
//-----------------------------------------------------------------------------------------------------------
   {
      Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> K((double*) intrinsics.data);
      pose(pts, train_g, query_g, depth, K, Q, translation);
   }

   void pose(const std::vector<std::pair<cv::Point3d, cv::Point3d>>& pts,
             const Eigen::Vector3d& train_g, const Eigen::Vector3d query_g, const double depth,
             const Eigen::Matrix3d& K, Eigen::Quaterniond& Q, Eigen::Vector3d& translation)
   //--------------------------------------------------------------------------------------
   {
      size_t m = pts.size();
//   if ( (! std::isnan(depth)) && (m > 6 ) )
//   {
//      pose_ransac(pts, train_g, query_g, depth, K, Q, translation, 3);
//      return;
//   }
      Eigen::Matrix3d KI = K.inverse();
      Eigen::Vector3d train_gravity(train_g), query_gravity(query_g);
      train_gravity.normalize();
      query_gravity.normalize();
//   Q = rotation(train_gravity, query_gravity); // camera to model
      Q = rotation(query_gravity, train_gravity); // model to camera

      Eigen::Matrix3d R = Q.toRotationMatrix();
      Eigen::MatrixXd A(3 * m, 3);
      Eigen::VectorXd b(m * 3);
      for (size_t row = 0, ri = 0; row < m; row++)
      {
         const cv::Point3d& tpt = pts[row].first;
         const cv::Point3d& qpt = pts[row].second;
         Eigen::Vector3d train_ray = KI*Eigen::Vector3d(tpt.x, tpt.y, 1);
         Eigen::Vector3d query_ray = KI*Eigen::Vector3d(qpt.x, qpt.y, 1);
         double xt1 = train_ray[0], yt1 = train_ray[1], xq1 = query_ray[0], yq1 = query_ray[1];
#ifdef USE_ROTATION_MATRIX
         A.row(ri) << 0, 1, -yq1;
         b(ri++) = b0(R, xt1, yt1, depth, yq1);
         A.row(ri) << -1, 0, xq1;
         b(ri++) = b1(R, xt1, yt1, depth, xq1);
         A.row(ri) << yq1, -xq1, 0;
         b(ri++) = b2(R, xt1, yt1, depth, xq1, yq1);
#endif
#ifdef USE_QUATERNION
         A.row(ri) << 0, -1, yq1;
         b(ri++) = b0(Q.w(), Q.x(), Q.y(), Q.z(), xt1, yt1, yq1, d);
         A.row(ri) <<  1, 0, -xq1;
         b(ri++) = b1(Q.w(), Q.x(), Q.y(), Q.z(), xt1, yt1, xq1, d);
         A.row(ri) << -yq1, xq1, 0;
         b(ri++) = b2(Q.w(), Q.x(), Q.y(), Q.z(), xt1, yt1, xq1, yq1, d);
#endif
      }
   std::cout << "Rank " << A.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).rank() << std::endl;
//      Eigen::ColPivHouseholderQR<Eigen::MatrixXd> MQR(A);
//      translation = MQR.solve(b);
   translation = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
   }

   double pose_ransac(const std::vector<std::pair<cv::Point3d, cv::Point3d>>& pts,
                      const Eigen::Vector3d& train_g, const Eigen::Vector3d query_g,
                      const cv::Mat& intrinsics, Eigen::Quaterniond& Q, Eigen::Vector3d& translation,
                      void* RANSAC_params, int samples)
   //------------------------------------------------------------------------------------------------
   {
      Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> K((double*) intrinsics.data);
      return pose_ransac(pts, train_g, query_g, K, Q, translation, RANSAC_params, samples);
   }

   double pose_ransac(const std::vector<std::pair<cv::Point3d, cv::Point3d>>& pts,
                      const Eigen::Vector3d& train_g, const Eigen::Vector3d query_g,
                      const Eigen::Matrix3d& K, Eigen::Quaterniond& Q, Eigen::Vector3d& translation,
                      void* RANSAC_params, int samples)
   //-----------------------------------------------------------------------------------------------
   {
      if (RANSAC_params == nullptr) throw std::logic_error("pose2d::pose_ransac (no depth): RANSAC params are null");
      Eigen::Vector3d train_gravity(train_g), query_gravity(query_g);
      train_gravity.normalize();
      query_gravity.normalize();
//   Q = rotation(train_gravity, query_gravity); // camera to model
      Q = rotation(query_gravity, train_gravity); // model to camera
      double confidence = -1;
#ifdef USE_THEIA_RANSAC
      Grav2DRansacEstimator estimator(K, Q, -1, samples);
      theia::RansacParameters* parameters = static_cast<theia::RansacParameters*>(RANSAC_params);
      theia::RansacSummary summary;
      std::unique_ptr<theia::SampleConsensusEstimator<Grav2DRansacEstimator>> ransac =
            theia::CreateAndInitializeRansacVariant(theia::RansacType::RANSAC, *parameters, estimator);
      if (ransac)
      {
         GravPoseRansacModel best_model;
         ransac->Estimate(pts, &best_model, &summary);
         confidence = summary.confidence;
         if (confidence > 0)
         {
            Q = best_model.rotation;
            translation = best_model.translation;
         }
      }
#else
      templransac::RANSACParams* parameters = static_cast<templransac::RANSACParams*>(RANSAC_params);
      Grav2DRansacEstimator estimator(K, Q);
      Grav2DRansacData data(pts);
      std::vector<std::pair<double, GravPoseRansacModel> > results;
      std::vector<std::vector<size_t>> inlier_indices;
      std::stringstream errs;
      confidence = templransac::RANSAC(*parameters, estimator, data, pts.size(), samples, 1,
                                       results, inlier_indices, &errs);
      if (confidence > 0)
      {
         GravPoseRansacModel& model = results[0].second;
         translation = model.translation;
      }
#endif
      return confidence;
   }

   double pose_ransac(const std::vector<std::pair<cv::Point3d, cv::Point3d>>& pts,
                      const Eigen::Vector3d& train_g, const Eigen::Vector3d query_g,
                      const double depth, const cv::Mat& intrinsics,
                      Eigen::Quaterniond& Q, Eigen::Vector3d& translation,
                      void* RANSAC_params, int samples)
   //----------------------------------------------------------------------------------------------------------
   {
      Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> K((double*) intrinsics.data);
      return pose_ransac(pts, train_g, query_g, depth, K, Q, translation, RANSAC_params, samples);
   }

   double pose_ransac(const std::vector<std::pair<cv::Point3d, cv::Point3d>>& pts,
                      const Eigen::Vector3d& train_g, const Eigen::Vector3d query_g,
                      const double depth, const Eigen::Matrix3d& K,
                      Eigen::Quaterniond& Q, Eigen::Vector3d& translation,
                      void *RANSAC_params, int samples)
   //---------------------------------------------------------------------------------
   {
      if (RANSAC_params == nullptr) throw std::logic_error("pose2d::pose_ransac (with depth): RANSAC params are null");
      Eigen::Vector3d train_gravity(train_g), query_gravity(query_g);
      train_gravity.normalize();
      query_gravity.normalize();
//   Q = rotation(train_gravity, query_gravity); // camera to model
      Q = rotation(query_gravity, train_gravity); // model to camera
      double confidence = -1;
#ifdef USE_THEIA_RANSAC
      Grav2DRansacEstimator estimator(K, Q, depth, samples);
      theia::RansacParameters* parameters = static_cast<theia::RansacParameters*>(RANSAC_params);
      theia::RansacSummary summary;
      std::unique_ptr<theia::SampleConsensusEstimator<Grav2DRansacEstimator>> ransac =
            theia::CreateAndInitializeRansacVariant(theia::RansacType::RANSAC, *parameters, estimator);
      if (ransac)
      {
         GravPoseRansacModel best_model;

         if (ransac->Estimate(pts, &best_model, &summary))
         {
            confidence = summary.confidence;
            if (confidence > 0)
            {
               Q = best_model.rotation;
               translation = best_model.translation;
            }
         }
      }
#else
      templransac::RANSACParams* parameters = static_cast<templransac::RANSACParams*>(RANSAC_params);
      Grav2DDepthRansacEstimator estimator(K, Q, depth);
      Grav2DRansacData data(pts);
      std::vector<std::pair<double, GravPoseRansacModel>> results;
      std::vector<std::vector<size_t>> inlier_indices;
      std::stringstream errs;
      confidence = templransac::RANSAC(*parameters, estimator, data, pts.size(), samples, 1,
                                       results, inlier_indices, &errs);
      if (confidence > 0)
      {
         GravPoseRansacModel& model = results[0].second;
         translation = model.translation;

//      for (size_t k=0; k<results.size(); k++)
//      {
//         std::pair<double, Grav2DRansacModel> pp = results[k];
//         std::cout << "RANSAC Result " << pp.second.translation.transpose() << " ";
//         std::vector<size_t> inliers = inlier_indices[k];
//         for (size_t inlier : inliers)
//            std::cout << train_img_pts[inlier] << " -> " << query_image_pts[inlier] << " | ";
//         std::cout << std::endl;
//      }
      }
#endif
      return confidence;
   }

   void pose_translation(const Eigen::Matrix3d& Kinv, const std::vector<std::pair<cv::Point3d, cv::Point3d>>& pts,
                         const Eigen::Matrix3d& R, Eigen::Vector3d& translation)
//----------------------------------------------------------------------------------------------------------------
   {
      const cv::Point3d& tpt0 = pts[0].first;
      const cv::Point3d& qpt0 = pts[0].second;
      const cv::Point3d& tpt1 = pts[1].first;
      const cv::Point3d& qpt1 = pts[1].second;
      const cv::Point3d& tpt2 = pts[2].first;
      const cv::Point3d& qpt2 = pts[2].second;
      Eigen::Vector3d train_ray1 = Kinv*Eigen::Vector3d(tpt0.x, tpt0.y, 1);
      Eigen::Vector3d query_ray1 = Kinv*Eigen::Vector3d(qpt0.x, qpt0.y, 1);
      Eigen::Vector3d train_ray2 = Kinv*Eigen::Vector3d(tpt1.x, tpt1.y, 1);
      Eigen::Vector3d query_ray2 = Kinv*Eigen::Vector3d(qpt1.x, qpt1.y, 1);
      Eigen::Vector3d train_ray3 = Kinv*Eigen::Vector3d(tpt2.x, tpt2.y, 1);
      Eigen::Vector3d query_ray3 = Kinv*Eigen::Vector3d(qpt2.x, qpt2.y, 1);
      double xt1 = train_ray1[0], yt1 = train_ray1[1], xq1 = query_ray1[0], yq1 = query_ray1[1];
      double xt2 = train_ray2[0], yt2 = train_ray2[1], xq2 = query_ray2[0], yq2 = query_ray2[1];
      double xt3 = train_ray3[0], yt3 = train_ray3[1], xq3 = query_ray3[0], yq3 = query_ray3[1];
      Eigen::Matrix3d A;
      A << tx_coeff(xt1, yt1, xq1, yq1, R),
            ty_coeff(xt1, yt1, xq1, yq1, R),
            tz_coeff(xt1, yt1, xq1, yq1, R),

            tx_coeff(xt2, yt2, xq2, yq2, R),
            ty_coeff(xt2, yt2, xq2, yq2, R),
            tz_coeff(xt2, yt2, xq2, yq2, R),

            tx_coeff(xt3, yt3, xq3, yq3, R),
            ty_coeff(xt3, yt3, xq3, yq3, R),
            tz_coeff(xt3, yt3, xq3, yq3, R);
      translation = homogeneous(A);
   }

   //Called by RANSAC estimation
   void pose_translation(const Eigen::Matrix3d& Kinv, const std::vector<cv::Point3d>& train_img_pts,
                         const std::vector<cv::Point3d>& query_img_pts,
                         const Eigen::Matrix3d& R, Eigen::Vector3d& translation)
//------------------------------------------------------------------------
   {
      const cv::Point3d& tpt0 = train_img_pts[0];
      const cv::Point3d& tpt1 = train_img_pts[1];
      const cv::Point3d& tpt2 = train_img_pts[2];
      const cv::Point3d& qpt0 = query_img_pts[0];
      const cv::Point3d& qpt1 = query_img_pts[1];
      const cv::Point3d& qpt2 = query_img_pts[2];
      Eigen::Vector3d train_ray1 = Kinv*Eigen::Vector3d(tpt0.x, tpt0.y, 1);
      Eigen::Vector3d query_ray1 = Kinv*Eigen::Vector3d(qpt0.x, qpt0.y, 1);
      Eigen::Vector3d train_ray2 = Kinv*Eigen::Vector3d(tpt1.x, tpt1.y, 1);
      Eigen::Vector3d query_ray2 = Kinv*Eigen::Vector3d(qpt1.x, qpt1.y, 1);
      Eigen::Vector3d train_ray3 = Kinv*Eigen::Vector3d(tpt2.x, tpt2.y, 1);
      Eigen::Vector3d query_ray3 = Kinv*Eigen::Vector3d(qpt2.x, qpt2.y, 1);
      double xt1 = train_ray1[0], yt1 = train_ray1[1], xq1 = query_ray1[0], yq1 = query_ray1[1];
      double xt2 = train_ray2[0], yt2 = train_ray2[1], xq2 = query_ray2[0], yq2 = query_ray2[1];
      double xt3 = train_ray3[0], yt3 = train_ray3[1], xq3 = query_ray3[0], yq3 = query_ray3[1];
      Eigen::Matrix3d A;
      A << tx_coeff(xt1, yt1, xq1, yq1, R),
            ty_coeff(xt1, yt1, xq1, yq1, R),
            tz_coeff(xt1, yt1, xq1, yq1, R),

            tx_coeff(xt2, yt2, xq2, yq2, R),
            ty_coeff(xt2, yt2, xq2, yq2, R),
            tz_coeff(xt2, yt2, xq2, yq2, R),

            tx_coeff(xt3, yt3, xq3, yq3, R),
            ty_coeff(xt3, yt3, xq3, yq3, R),
            tz_coeff(xt3, yt3, xq3, yq3, R);
      translation = homogeneous(A);
   }

   void pose_translation(const Eigen::Matrix3d& Kinv, const std::vector<std::pair<cv::Point3d, cv::Point3d>>& pts, const double depth,
                         const Eigen::Matrix3d& R, Eigen::Vector3d& translation)
   //-------------------------------------------------------------------------
   {
      const cv::Point3d& tpt0 = pts[0].first;
      const cv::Point3d& qpt0 = pts[0].second;
      const cv::Point3d& tpt1 = pts[1].first;
      const cv::Point3d& qpt1 = pts[1].second;
      const cv::Point3d& tpt2 = pts[2].first;
      const cv::Point3d& qpt2 = pts[2].second;
      Eigen::Vector3d train_ray1 = Kinv*Eigen::Vector3d(tpt0.x, tpt0.y, 1);
      Eigen::Vector3d query_ray1 = Kinv*Eigen::Vector3d(qpt0.x, qpt0.y, 1);
      Eigen::Vector3d train_ray2 = Kinv*Eigen::Vector3d(tpt1.x, tpt1.y, 1);
      Eigen::Vector3d query_ray2 = Kinv*Eigen::Vector3d(qpt1.x, qpt1.y, 1);
      Eigen::Vector3d train_ray3 = Kinv*Eigen::Vector3d(tpt2.x, tpt2.y, 1);
      Eigen::Vector3d query_ray3 = Kinv*Eigen::Vector3d(qpt2.x, qpt2.y, 1);
      double xt1 = train_ray1[0], yt1 = train_ray1[1], xq1 = query_ray1[0], yq1 = query_ray1[1];
      double xt2 = train_ray2[0], yt2 = train_ray2[1], xq2 = query_ray2[0], yq2 = query_ray2[1];
      double xt3 = train_ray3[0], yt3 = train_ray3[1], xq3 = query_ray3[0], yq3 = query_ray3[1];
//   Eigen::MatrixXd A(9, 3);
//   Eigen::VectorXd b(9);
      Eigen::Matrix<double, 9, 3> A;
      Eigen::Matrix<double, 9, 1> b;
      A << 0, 1, -yq1,
            -1, 0, xq1,
            yq1, -xq1, 0,

            0, 1, -yq2,
            -1, 0, xq2,
            yq2, -xq2, 0,

            0, 1, -yq3,
            -1, 0, xq3,
            yq3, -xq3, 0;

      b << b0(R, xt1, yt1, depth, yq1), b1(R, xt1, yt1, depth, xq1), b2(R, xt1, yt1, depth, xq1, yq1),
            b0(R, xt2, yt2, depth, yq2), b1(R, xt2, yt2, depth, xq2), b2(R, xt2, yt2, depth, xq2, yq2),
            b0(R, xt3, yt3, depth, yq3), b1(R, xt3, yt3, depth, xq3), b2(R, xt3, yt3, depth, xq3, yq3);
//      Eigen::ColPivHouseholderQR<Eigen::Matrix<double, 9, 3>> MQR(A);
//      translation = MQR.solve(b);
      translation = A.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).solve(b);
//   std::cout << "Rank " << A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).rank() << std::endl;
   }

   //Called by RANSAC estimation
   void pose_translation(const Eigen::Matrix3d& Kinv, const std::vector<cv::Point3d>& train_img_pts,
                         const std::vector<cv::Point3d>& query_img_pts, const double depth,
                         const Eigen::Matrix3d& R, Eigen::Vector3d& translation)
   //-------------------------------------------------------------------------
   {
      const cv::Point3d& tpt0 = train_img_pts[0];
      const cv::Point3d& tpt1 = train_img_pts[1];
      const cv::Point3d& tpt2 = train_img_pts[2];
      const cv::Point3d& qpt0 = query_img_pts[0];
      const cv::Point3d& qpt1 = query_img_pts[1];
      const cv::Point3d& qpt2 = query_img_pts[2];
      Eigen::Vector3d train_ray1 = Kinv*Eigen::Vector3d(tpt0.x, tpt0.y, 1);
      Eigen::Vector3d query_ray1 = Kinv*Eigen::Vector3d(qpt0.x, qpt0.y, 1);
      Eigen::Vector3d train_ray2 = Kinv*Eigen::Vector3d(tpt1.x, tpt1.y, 1);
      Eigen::Vector3d query_ray2 = Kinv*Eigen::Vector3d(qpt1.x, qpt1.y, 1);
      Eigen::Vector3d train_ray3 = Kinv*Eigen::Vector3d(tpt2.x, tpt2.y, 1);
      Eigen::Vector3d query_ray3 = Kinv*Eigen::Vector3d(qpt2.x, qpt2.y, 1);
      double xt1 = train_ray1[0], yt1 = train_ray1[1], xq1 = query_ray1[0], yq1 = query_ray1[1];
      double xt2 = train_ray2[0], yt2 = train_ray2[1], xq2 = query_ray2[0], yq2 = query_ray2[1];
      double xt3 = train_ray3[0], yt3 = train_ray3[1], xq3 = query_ray3[0], yq3 = query_ray3[1];
//   Eigen::MatrixXd A(9, 3);
//   Eigen::VectorXd b(9);
      Eigen::Matrix<double, 9, 3> A;
      Eigen::Matrix<double, 9, 1> b;
      A << 0, 1, -yq1,
            -1, 0, xq1,
            yq1, -xq1, 0,

            0, 1, -yq2,
            -1, 0, xq2,
            yq2, -xq2, 0,

            0, 1, -yq3,
            -1, 0, xq3,
            yq3, -xq3, 0;

      b << b0(R, xt1, yt1, depth, yq1), b1(R, xt1, yt1, depth, xq1), b2(R, xt1, yt1, depth, xq1, yq1),
            b0(R, xt2, yt2, depth, yq2), b1(R, xt2, yt2, depth, xq2), b2(R, xt2, yt2, depth, xq2, yq2),
            b0(R, xt3, yt3, depth, yq3), b1(R, xt3, yt3, depth, xq3), b2(R, xt3, yt3, depth, xq3, yq3);
//      Eigen::ColPivHouseholderQR<Eigen::Matrix<double, 9, 3>> MQR(A);
//      translation = MQR.solve(b);
      translation = A.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).solve(b);
//   std::cout << "Rank " << A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).rank() << std::endl;
   }
}