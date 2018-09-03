#include <iostream>
#include <vector>
#include <chrono>

//#define _RANSAC_STATS_ 1
#ifdef _RANSAC_STATS_
#include <limits>
#endif

#include <Eigen/src/Core/util/DisableStupidWarnings.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <Eigen/Dense>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>

#include <opencv2/core/eigen.hpp>

#include "pose3d.h"
#include "Optimization.h"
#include "Ransac.hh"

#define USE_ROTATION_MATRIX
//#define USE_QUATERNION

#define USE_SVD
//#define USE_QR

namespace pose3d
{
   inline double mrhs1(double X_1, double Y_1, double Z_1, double y_1, double r_10, double r_11, double r_12,
                       double r_20, double r_21, double r_22)
//---------------------------------------------------------------------------------------------------------
   {
      return X_1 * r_10 - X_1 * r_20 * y_1 + Y_1 * r_11 - Y_1 * r_21 * y_1 + Z_1 * r_12 - Z_1 * r_22 * y_1;
   }

   inline double mrhs2(double X_1, double Y_1, double Z_1, double x_1,
                       double r_00, double r_01, double r_02, double r_20, double r_21, double r_22)
//------------------------------------------------------------------------------------------------
   {
      return -X_1 * r_00 + X_1 * r_20 * x_1 - Y_1 * r_01 + Y_1 * r_21 * x_1 - Z_1 * r_02 + Z_1 * r_22 * x_1;
   }

   inline double mrhs3(double X_1, double Y_1, double Z_1, double x_1, double y_1,
                       double r_00, double r_01, double r_02, double r_10, double r_11, double r_12)
//-----------------------------------------------------------------------------------------------
   {
      return X_1 * r_00 * y_1 - X_1 * r_10 * x_1 + Y_1 * r_01 * y_1 - Y_1 * r_11 * x_1 + Z_1 * r_02 * y_1 -
             Z_1 * r_12 * x_1;
   }

   inline double qrhs1(double Qw, double Qx, double Qy, double Qz, double X_1, double Y_1, double Z_1, double y_1)
//-------------------------------------------------------------------------------------------------------------
   {
      double Qw2 = Qw * Qw, Qx2 = Qx * Qx, Qy2 = Qy * Qy, Qz2 = Qz * Qz, Qwx = Qw * Qx;
      return -Qw2 * Z_1 * y_1 - 2 * Qwx * Y_1 * y_1 + 2 * Qw * Qy * X_1 * y_1 + Qx2 * Z_1 * y_1 -
             2 * Qx * Qz * X_1 * y_1 +
             Qy2 * Z_1 * y_1 - 2 * Qy * Qz * Y_1 * y_1 - Qz2 * Z_1 * y_1 + 2 * X_1 * (Qw * Qz + Qx * Qy) + Y_1 * (Qw2 -
                                                                                                                  Qx2 +
                                                                                                                  Qy2 -
                                                                                                                  Qz2) -
             2 * Z_1 * (Qwx - Qy * Qz);
   }

   inline double qrhs2(double Qw, double Qx, double Qy, double Qz, double X_1, double Y_1, double Z_1, double x_1)
//------------------------------------------------------------------------------------------------------------
   {
      double Qx2 = Qx * Qx, Qy2 = Qy * Qy, Qz2 = Qz * Qz, Qwy = Qw * Qy;
      return Qw * Qw * Z_1 * x_1 + 2 * Qw * Qx * Y_1 * x_1 - 2 * Qwy * X_1 * x_1 - Qx2 * Z_1 * x_1 +
             2 * Qx * Qz * X_1 * x_1 -
             Qy2 * Z_1 * x_1 + 2 * Qy * Qz * Y_1 * x_1 + Qz2 * Z_1 * x_1 - X_1 * (Qw * Qw + Qx2 -
                                                                                  Qy2 - Qz2) -
             2 * Y_1 * (-Qw * Qz + Qx * Qy) - 2 * Z_1 * (Qwy + Qx * Qz);
   }

   inline double
   qrhs3(double Qw, double Qx, double Qy, double Qz, double X_1, double Y_1, double Z_1, double x_1, double y_1)
//-----------------------------------------------------------------------------------------------------------------
   {
      double Qw2 = Qw * Qw, Qx2 = Qx * Qx, Qy2 = Qy * Qy, Qz2 = Qz * Qz, Qwz = Qw * Qz;
      return Qw2 * X_1 * y_1 - Qw2 * Y_1 * x_1 + 2 * Qw * Qx * Z_1 * x_1 + 2 * Qw * Qy * Z_1 * y_1 -
             2 * Qwz * X_1 * x_1 -
             2 * Qwz * Y_1 * y_1 + Qx2 * X_1 * y_1 + Qx2 * Y_1 * x_1 - 2 * Qx * Qy * X_1 * x_1 +
             2 * Qx * Qy * Y_1 * y_1 +
             2 * Qx * Qz * Z_1 * y_1 - Qy2 * X_1 * y_1 - Qy2 * Y_1 * x_1 - 2 * Qy * Qz * Z_1 * x_1 - Qz2 * X_1 * y_1 +
             Qz2 * Y_1 * x_1;
   }

   inline double dotg(double X_1, double Y_1, double Z_1, double X_2, double Y_2, double Z_2,
                      double g_x, double g_y, double g_z, const Eigen::Matrix3d& R)
   {
      return g_x * (X_1 * R(0, 0) - X_2 * R(0, 0) + Y_1 * R(0, 1) - Y_2 * R(0, 1) + Z_1 * R(0, 2) - Z_2 * R(0, 2)) +
             g_y * (X_1 * R(1, 0) - X_2 * R(1, 0) +
                    Y_1 * R(1, 1) - Y_2 * R(1, 1) + Z_1 * R(1, 2) - Z_2 * R(1, 2)) +
             g_z * (X_1 * R(2, 0) - X_2 * R(2, 0) + Y_1 * R(2, 1) - Y_2 * R(2, 1) +
                    Z_1 * R(2, 2) - Z_2 * R(2, 2));
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

#ifdef USE_ROTATION_MATRIX
   bool pose(const std::vector<std::pair<cv::Point3d, cv::Point2d>>& pts,
             const Eigen::Vector3d& train_g, const Eigen::Vector3d query_g,
             const Eigen::Matrix3d& KI, Eigen::Quaterniond& Q, Eigen::Vector3d& translation)
//-----------------------------------------------------------------------------------------------------------
   {
      const size_t m = pts.size();
//      if (m > 3)
//      {
//         std::cerr << "Use pose_ransac for more than 3 points" << std::endl;
//         // pose_ransac(world_pts, image_pts, train_g, query_g, KI, Q, translation);
//         return false;
//      }
      Eigen::Vector3d train_gravity(train_g), query_gravity(query_g);
      train_gravity.normalize();
      query_gravity.normalize();
//   Q = rotation(train_gravity, query_gravity); // camera to model
      Q = rotation(query_gravity, train_gravity); // model to camera

      Eigen::Matrix3d R = Q.toRotationMatrix();
      const cv::Point3d &world_pt0 = pts[0].first, &world_pt1 = pts[1].first,
                        &world_pt2 = pts[2].first;
      const cv::Point2d &image_pt0 = pts[0].second, &image_pt1 = pts[1].second,
                        &image_pt2 = pts[2].second;
      Eigen::Vector3d query_ray1 = KI*Eigen::Vector3d(image_pt0.x, image_pt0.y, 1),
                      query_ray2 = KI*Eigen::Vector3d(image_pt1.x, image_pt1.y, 1),
                      query_ray3 = KI*Eigen::Vector3d(image_pt2.x, image_pt2.y, 1);
      double Xt1 = world_pt0.x, Yt1 = world_pt0.y, Zt1 = world_pt0.z,
             Xt2 = world_pt1.x, Yt2 = world_pt1.y, Zt2 = world_pt1.z,
             Xt3 = world_pt2.x, Yt3 = world_pt2.y, Zt3 = world_pt2.z,
             xq1 = query_ray1[0], yq1 = query_ray1[1], xq2 = query_ray2[0], yq2 = query_ray2[1],
             xq3 = query_ray3[0], yq3 = query_ray3[1];

      Eigen::Matrix<double, 9, 3> A;
      Eigen::Matrix<double, 9, 1> b;
//   Eigen::Matrix<double, 6, 3> A;
//   Eigen::Matrix<double, 6, 1> b;
      A << 0, -1, yq1,
            1, 0, -xq1,
            -yq1, xq1, 0,
            0, -1, yq2,
            1, 0, -xq2,
            -yq2, xq2, 0,
            0, -1, yq3,
            1, 0, -xq3,
            -yq3, xq3, 0;
      double r_00 = R(0, 0), r_01 = R(0, 1), r_02 = R(0, 2),
             r_10 = R(1, 0), r_11 = R(1, 1), r_12 = R(1, 2),
             r_20 = R(2, 0), r_21 = R(2, 1), r_22 = R(2, 2);
      b <<  mrhs1(Xt1, Yt1, Zt1, yq1, r_10, r_11, r_12, r_20, r_21, r_22),
            mrhs2(Xt1, Yt1, Zt1, xq1, r_00, r_01, r_02, r_20, r_21, r_22),
            mrhs3(Xt1, Yt1, Zt1, xq1, yq1, r_00, r_01, r_02, r_10, r_11, r_12),
            mrhs1(Xt2, Yt2, Zt2, yq2, r_10, r_11, r_12, r_20, r_21, r_22),
            mrhs2(Xt2, Yt2, Zt2, xq2, r_00, r_01, r_02, r_20, r_21, r_22),
            mrhs3(Xt2, Yt2, Zt2, xq2, yq2, r_00, r_01, r_02, r_10, r_11, r_12),
            mrhs1(Xt3, Yt3, Zt3, yq3, r_10, r_11, r_12, r_20, r_21, r_22),
            mrhs2(Xt3, Yt3, Zt3, xq3, r_00, r_01, r_02, r_20, r_21, r_22),
            mrhs3(Xt3, Yt3, Zt3, xq3, yq3, r_00, r_01, r_02, r_10, r_11, r_12);
//      std::cout << A << std::endl << "Rank " << A.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).rank() << std::endl;
#ifdef USE_SVD
      translation = A.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).solve(b);
#endif
#ifdef USE_QR
      Eigen::ColPivHouseholderQR<Eigen::Matrix<double, 9, 3>> MQR(A);
      translation = MQR.solve(b);
#endif

//   Eigen::MatrixXd A(m*3, 3);
//   Eigen::VectorXd b(m*3);
//    for (size_t row=0, ri=0; row<m; row++)
//    {
//       if (row == 3)
//       {
//          std::cout << row << " " << ri << std::endl;
//          break;
//       }

//       cv::Point2d& pt = const_cast<cv::Point2d &>(image_pts[row]);
//       Eigen::Vector3d ray = KI*Eigen::Vector3d(pt.x, pt.y, 1);
//       const double Xt = world_pts[row].x, Yt = world_pts[row].y, Zt = world_pts[row].z, xq = ray[0], yq = ray[1];

//       A(ri, 0) = 0;
//       A(ri, 1) = -1;
//       A(ri, 2) = yq;
//       b[ri++] = mrhs1(Xt, Yt, Zt, yq, r_10, r_11,  r_12, r_20, r_21, r_22);

//       A(ri, 0) = 1.0;
//       A(ri, 1) = 0.0;
//       A(ri, 2) = -xq;
//       b[ri++] = mrhs2(Xt, Yt, Zt, xq, r_00, r_01, r_02, r_20, r_21, r_22);

//       A(ri, 0) = -yq;
//       A(ri, 1) = xq;
//       A(ri, 2) = 0;
//       b[ri++] = mrhs3(Xt, Yt, Zt, xq, yq, r_00, r_01, r_02, r_10, r_11, r_12);
//    }
// //   std::cout << A << std::endl << b << std::endl;
//    translation = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
      return true;
   }

#endif

#ifdef USE_QUATERNION
   bool pose(const std::vector<std::pair<cv::Point3d, cv::Point2d>>& pts,
             const Eigen::Vector3d& train_g, const Eigen::Vector3d query_g,
             const Eigen::Matrix3d& KI, Eigen::Quaterniond& Q, Eigen::Vector3d& translation)
   //-----------------------------------------------------------------------------------------------------------
   {
      const size_t m = pts.size();
      if (m > 3)
      {
         std::cout << "Use pose_ransac for more than 3 points" << std::endl;
         return false;
      }
      Eigen::Vector3d train_gravity(train_g), query_gravity(query_g);
      train_gravity.normalize();
      query_gravity.normalize();
   //   Q = rotation(train_gravity, query_gravity); // camera to model
      Q = rotation(query_gravity, train_gravity); // model to camera
      std::vector<cv::Point3d> world_pts;
      std::vector<cv::Point2d> image_pts;
      for (const std::pair<cv::Point3d, cv::Point2d>& pp : pts)
      {
         const cv::Point3d &wpt = pp.first;
         const cv::Point2d &ipt = pp.second;
         world_pts.emplace_back(wpt.x, wpt.y, wpt.z);
         image_pts.emplace_back(ipt.x, ipt.y);
      }
      pose3d::pose_translation(world_pts, image_pts, KI, Q, translation);
   /*
      Eigen::MatrixXd A(m*3, 3);
      Eigen::VectorXd b(m*3);
   //   Eigen::MatrixXd A(m*2, 3);
   //   Eigen::VectorXd b(m*2);
      const double Qw = Q.w(), Qx = Q.x(), Qy = Q.y(), Qz = Q.z();
      for (size_t row=0, ri=0; row<m; row++)
      {
         cv::Point2d& pt = const_cast<cv::Point2d &>(image_pts[row]);
         Eigen::Vector3d ray = KI*Eigen::Vector3d(pt.x, pt.y, 1);
         const double Xt = world_pts[row].x, Yt = world_pts[row].y, Zt = world_pts[row].z, xq = ray[0], yq = ray[1];

         A(ri, 0) = 0;
         A(ri, 1) = -1;
         A(ri, 2) = yq;
         b[ri++] = qrhs1(Qw, Qx, Qy, Qz, Xt, Yt, Zt, xq);

         A(ri, 0) = 1.0;
         A(ri, 1) = 0.0;
         A(ri, 2) = -xq;
         b[ri++] = qrhs2(Qw, Qx, Qy, Qz, Xt, Yt, Zt, xq);

         A(ri, 0) = -yq;
         A(ri, 1) = xq;
         A(ri, 2) = 0;
         b[ri++] = qrhs3(Qw, Qx, Qy, Qz, Xt, Yt, Zt, xq, yq);
      }
      translation = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
   */
      return true;
   }
#endif

   void refine(const std::vector<cv::Point3d>& world_pts, const std::vector<cv::Point2d>& image_pts,
               const Eigen::Matrix3d& K, Eigen::Quaterniond& Q, Eigen::Vector3d& translation, int& iterations)
//---------------------------------------------------------------------------------------------------
   {
      translation_levenberg_marquardt3d(world_pts, image_pts, K, Q, translation, iterations);
   }

   double pose_ransac(const std::vector<std::pair<cv::Point3d, cv::Point2d>>& pts,
                      const Eigen::Vector3d& train_g, const Eigen::Vector3d query_g,
                      const Eigen::Matrix3d& KI, Eigen::Quaterniond& Q, Eigen::Vector3d& translation,
                      void *RANSAC_params, int samples)
//----------------------------------------------------------------------------------------------------------
   {
      if (RANSAC_params == nullptr) throw std::logic_error("pose3d::pose_ransac: RANSAC params are null");
      Eigen::Vector3d train_gravity(train_g), query_gravity(query_g);
      train_gravity.normalize();
      query_gravity.normalize();
      //   Q = rotation(train_gravity, query_gravity); // camera to model
      Q = rotation(query_gravity, train_gravity); // model to camera
      double confidence = -1;
#ifdef USE_THEIA_RANSAC
      Grav3DRansacEstimator estimator(KI, Q, samples);
      theia::RansacParameters* parameters = static_cast<theia::RansacParameters*>(RANSAC_params);
      theia::RansacSummary summary;
      std::unique_ptr<theia::SampleConsensusEstimator<Grav3DRansacEstimator>> ransac =
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
      Grav3DRansacEstimator estimator(KI, Q);
      Grav3DRansacData data(pts);
      std::vector<std::pair<double, GravPoseRansacModel> > results;
      std::vector<std::vector<size_t>> inlier_indices;
      std::stringstream errs;
      confidence = templransac::RANSAC(*parameters, estimator, data, pts.size(), samples, 1, results,
                                        inlier_indices, &errs);
      if (confidence > 0)
      {
         GravPoseRansacModel& model = results[0].second;
         translation = model.translation;
      }
#endif
      return confidence;
   }

   //Called by RANSAC estimation
   void pose_translation(const std::vector<cv::Point3d>& world_pts, const std::vector<cv::Point2d>& query_image_pts,
                         const Eigen::Matrix3d& KI, const Eigen::Matrix3d& R, Eigen::Vector3d& translation)
   //-----------------------------------------------------------------------------------------------------------
   {
      const cv::Point2d &qpt0 = query_image_pts[0], &qpt1 = query_image_pts[1], &qpt2 = query_image_pts[2];
      Eigen::Vector3d query_ray1 = KI*Eigen::Vector3d(qpt0.x, qpt0.y, 1),
            query_ray2 = KI*Eigen::Vector3d(qpt1.x, qpt1.y, 1),
            query_ray3 = KI*Eigen::Vector3d(qpt2.x, qpt2.y, 1);
      double Xt1 = world_pts[0].x, Yt1 = world_pts[0].y, Zt1 = world_pts[0].z,
            Xt2 = world_pts[1].x, Yt2 = world_pts[1].y, Zt2 = world_pts[1].z,
            Xt3 = world_pts[2].x, Yt3 = world_pts[2].y, Zt3 = world_pts[2].z,
            xq1 = query_ray1[0], yq1 = query_ray1[1], xq2 = query_ray2[0], yq2 = query_ray2[1],
            xq3 = query_ray3[0], yq3 = query_ray3[1];

      Eigen::Matrix<double, 9, 3> A;
      Eigen::Matrix<double, 9, 1> b;
//   Eigen::Matrix<double, 6, 3> A;
//   Eigen::Matrix<double, 6, 1> b;
      A << 0, -1, yq1,
            1, 0, -xq1,
            -yq1, xq1, 0,
            0, -1, yq2,
            1, 0, -xq2,
            -yq2, xq2, 0,
            0, -1, yq3,
            1, 0, -xq3,
            -yq3, xq3, 0;
      double r_00 = R(0, 0), r_01 = R(0, 1), r_02 = R(0, 2),
            r_10 = R(1, 0), r_11 = R(1, 1), r_12 = R(1, 2),
            r_20 = R(2, 0), r_21 = R(2, 1), r_22 = R(2, 2);
      b << mrhs1(Xt1, Yt1, Zt1, yq1, r_10, r_11, r_12, r_20, r_21, r_22),
            mrhs2(Xt1, Yt1, Zt1, xq1, r_00, r_01, r_02, r_20, r_21, r_22),
            mrhs3(Xt1, Yt1, Zt1, xq1, yq1, r_00, r_01, r_02, r_10, r_11, r_12),
            mrhs1(Xt2, Yt2, Zt2, yq2, r_10, r_11, r_12, r_20, r_21, r_22),
            mrhs2(Xt2, Yt2, Zt2, xq2, r_00, r_01, r_02, r_20, r_21, r_22),
            mrhs3(Xt2, Yt2, Zt2, xq2, yq2, r_00, r_01, r_02, r_10, r_11, r_12),
            mrhs1(Xt3, Yt3, Zt3, yq3, r_10, r_11, r_12, r_20, r_21, r_22),
            mrhs2(Xt3, Yt3, Zt3, xq3, r_00, r_01, r_02, r_20, r_21, r_22),
            mrhs3(Xt3, Yt3, Zt3, xq3, yq3, r_00, r_01, r_02, r_10, r_11, r_12);
//   std::cout << "Rank " << A.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).rank() << std::endl;
#ifdef USE_SVD
      translation = A.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).solve(b);
#endif
#ifdef USE_QR
      Eigen::ColPivHouseholderQR<Eigen::Matrix<double, 9, 3>> MQR(A);
      translation = MQR.solve(b);
#endif
   }

   //Called by RANSAC estimation
   void pose_translation(const std::vector<cv::Point3d>& world_pts, const std::vector<cv::Point2d>& query_image_pts,
                         const Eigen::Matrix3d& KI, const Eigen::Quaterniond& Q, Eigen::Vector3d& translation)
//-----------------------------------------------------------------------------------------------------------
   {
      const cv::Point2d &qpt0 = query_image_pts[0], &qpt1 = query_image_pts[1], &qpt2 = query_image_pts[2];
      Eigen::Vector3d query_ray1 = KI*Eigen::Vector3d(qpt0.x, qpt0.y, 1),
            query_ray2 = KI*Eigen::Vector3d(qpt1.x, qpt1.y, 1),
            query_ray3 = KI*Eigen::Vector3d(qpt2.x, qpt2.y, 1);
      double Xt1 = world_pts[0].x, Yt1 = world_pts[0].y, Zt1 = world_pts[0].z,
            Xt2 = world_pts[1].x, Yt2 = world_pts[1].y, Zt2 = world_pts[1].z,
            Xt3 = world_pts[2].x, Yt3 = world_pts[2].y, Zt3 = world_pts[2].z,
            xq1 = query_ray1[0], yq1 = query_ray1[1], xq2 = query_ray2[0], yq2 = query_ray2[1],
            xq3 = query_ray3[0], yq3 = query_ray3[1];

      Eigen::Matrix<double, 9, 3> A;
      Eigen::Matrix<double, 9, 1> b;
//   Eigen::Matrix<double, 6, 3> A;
//   Eigen::Matrix<double, 6, 1> b;
      A << 0, -1, yq1,
            1, 0, -xq1,
            -yq1, xq1, 0,
            0, -1, yq2,
            1, 0, -xq2,
            -yq2, xq2, 0,
            0, -1, yq3,
            1, 0, -xq3,
            -yq3, xq3, 0;

      double Qw = Q.w(), Qx = Q.x(), Qy = Q.y(), Qz = Q.z();
      b << qrhs1(Qw, Qx, Qy, Qz, Xt1, Yt1, Zt1, yq1),
            qrhs2(Qw, Qx, Qy, Qz, Xt1, Yt1, Zt1, xq1),
            qrhs3(Qw, Qx, Qy, Qz, Xt1, Yt1, Zt1, xq1, yq1),
            qrhs1(Qw, Qx, Qy, Qz, Xt2, Yt2, Zt2, yq2),
            qrhs2(Qw, Qx, Qy, Qz, Xt2, Yt2, Zt2, xq2),
            qrhs3(Qw, Qx, Qy, Qz, Xt2, Yt2, Zt2, xq2, yq2),
            qrhs1(Qw, Qx, Qy, Qz, Xt3, Yt3, Zt3, yq3),
            qrhs2(Qw, Qx, Qy, Qz, Xt3, Yt3, Zt3, xq3),
            qrhs3(Qw, Qx, Qy, Qz, Xt3, Yt3, Zt3, xq3, yq3);

//   std::cout << "Rank " << A.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).rank() << std::endl;
#ifdef USE_SVD
      translation = A.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).solve(b);
#endif
#ifdef USE_QR
      Eigen::ColPivHouseholderQR<Eigen::Matrix<double, 9, 3>> MQR(A);
      translation = MQR.solve(b);
#endif
   }
}