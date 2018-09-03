#include <iostream>

#include <opencv2/calib3d.hpp>
#include <opencv2/video/tracking.hpp>

#include "pose-other.h"
#include "math.hh"
#include "display.h"
#include "five_point_relative_pose.h"
#include "two_point_pose_partial_rotation.h"

namespace poseother
{
   void essential_pose(const cv::Mat& intrinsics, const std::vector<cv::Point3d>& train_img_pts,
                       const std::vector<cv::Point3d>& query_img_pts)
//------------------------------------------------------------------------------------------------------------
   {
      std::vector<cv::Point2d> train_pts, query_pts;
      for (cv::Point3d pt : train_img_pts)
         train_pts.emplace_back(pt.x, pt.y);
      for (cv::Point3d pt : query_img_pts)
         query_pts.emplace_back(pt.x, pt.y);
      cv::Mat Es = cv::findEssentialMat(train_pts, query_pts, intrinsics);
      std::cout << Es << std::endl << "--00000--" << std::endl;
      auto no = Es.rows / 3;
      for (auto i = 0; i < no; i++)
      {
         cv::Mat E(Es, cv::Rect(0, i * 3, 3, 3));
         std::cout << E << std::endl << "--00000--" << std::endl;
         cv::Mat R1, R2, T;
         cv::decomposeEssentialMat(E, R1, R2, T);
         cv::Vec3d euler = mut::rotation2Euler(R1);
         std::cout << "Raw Essential" << std::endl
                   << "-------------------------------------------------------------" << std::endl;
         std::cout << "Rotation1: Roll " << mut::radiansToDegrees(euler[0]) << " (" << euler[0] << "), Pitch "
                   << mut::radiansToDegrees(euler[1]) << " (" << euler[1] << ") Yaw " << mut::radiansToDegrees(euler[2])
                   << " (" << euler[2] << ")" << std::endl;
         euler = mut::rotation2Euler(R2);
         std::cout << "Rotation2: Roll " << mut::radiansToDegrees(euler[0]) << " (" << euler[0] << "), Pitch "
                   << mut::radiansToDegrees(euler[1]) << " (" << euler[1] << ") Yaw " << mut::radiansToDegrees(euler[2])
                   << " (" << euler[2] << ")" << std::endl;
         std::cout << "Translation1 " << T.t() << std::endl;

         cv::Mat R3, T3;
         cv::recoverPose(E, train_pts, query_pts, R3, T3);
         cv::Vec3d euler2 = mut::rotation2Euler(R3);
         std::cout << "Rotation: Roll " << mut::radiansToDegrees(euler[0]) << " (" << euler[0] << "), Pitch "
                   << mut::radiansToDegrees(euler[1]) << " (" << euler[1] << ") Yaw " << mut::radiansToDegrees(euler[2])
                   << " (" << euler[2] << ")" << std::endl;
         std::cout << "Translation " << T3.t() << std::endl;
      }
      std::cout << "=======================================================================" << std::endl;
   }

   void FivePointRelativePose(const cv::Mat& intrinsics, const std::vector<cv::Point3d>& train_img_pts,
                              const std::vector<cv::Point3d>& query_img_pts,
                              std::vector<Eigen::Quaterniond>& Qs, std::vector<Eigen::Vector3d>& Ts)
//-----------------------------------------------------------------------------------------------------------
   {
      std::vector<Eigen::Vector2d> image1_points, image2_points;
      Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> K((double*) intrinsics.data);
      Eigen::Matrix3d KI = K.inverse();
      for (int i = 0; i < 5; i++)
      {
         cv::Point3d pt = train_img_pts[i];
         Eigen::Vector3d ipt = KI * Eigen::Vector3d(pt.x, pt.y, pt.z);
         ipt.normalize();
         ipt = ipt / ipt[2];
         image1_points.emplace_back(ipt.x(), ipt.y());
         pt = query_img_pts[i];
         ipt = KI * Eigen::Vector3d(pt.x, pt.y, pt.z);
         ipt.normalize();
         ipt = ipt / ipt[2];
         image2_points.emplace_back(ipt.x(), ipt.y());
      }
      std::vector<Eigen::Matrix3d> essentials;
      Eigen::Matrix3d R1, R2;
      Eigen::Vector3d T;
      ::FivePointRelativePose(image1_points, image2_points, &essentials);
      if (essentials.size() > 0)
      {
         Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> EK((double*) intrinsics.data);
         int ino = 1;
         for (Eigen::Matrix3d& E :essentials)
         {
            cv::Mat cvE, cvR1, cvR2, cvT;
            cv::eigen2cv(E, cvE);
            cv::decomposeEssentialMat(cvE, cvR1, cvR2, cvT);
            Eigen::Quaterniond Q1, Q2;
            Eigen::Vector3d T;
            if (! cvR1.empty())
            {
               Eigen::Matrix3d R;
               cv::cv2eigen(cvR1, R);
               Q1 = Eigen::Quaterniond(R);
            }
            if (! cvR2.empty())
            {
               Eigen::Matrix3d R;
               cv::cv2eigen(cvR2, R);
               Q2 = Eigen::Quaterniond(R);
            }
            T[0] = cvT.at<double>(0, 0);
            T[1] = cvT.at<double>(1, 0);
            T[2] = cvT.at<double>(2, 0);
            Qs.push_back(Q1); Ts.push_back(T);
            Qs.push_back(Q2); Ts.push_back(T);
         }
      }
   }

   //Sweeney et al (http://cs.ucsb.edu/~bnuernberger/ismar2015.pdf)
   void TwoPointPosePartialRotation(const cv::Mat& intrinsics, const std::vector<cv::Point3d>& train_pts,
                                    const std::vector<cv::Point3d>& image_pts,
                                    const cv::Vec3f query_g, const cv::Vec3f down_axis,
                                    std::vector<Eigen::Quaterniond>& Qs, std::vector<Eigen::Vector3d>& Ts)
   //--------------------------------------------------------------------------------------------------------
   {
      if ((train_pts.size() < 2) || (image_pts.size() < 2)) return;
      Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> EK((double*) intrinsics.data);
      Eigen::Matrix3d EKI = EK.inverse();
      Eigen::Vector3d Yaxis(down_axis[0], down_axis[1], down_axis[2]);
      Yaxis.normalize();
      Eigen::Vector3d query_gravity(query_g[0], query_g[1], query_g[2]);
      Eigen::Quaterniond Q = Eigen::Quaterniond::FromTwoVectors(query_gravity.normalized(), Yaxis);
      Eigen::Matrix3d R = Q.toRotationMatrix();
      const cv::Point3d& tp1 = train_pts[0], & tp2 = train_pts[1], & qp1 = image_pts[0], & qp2 = image_pts[1];
      Eigen::Vector3d tpt1(tp1.x, tp1.y, tp1.z), tpt2(tp2.x, tp2.y, tp2.z),
            qpt1 = R * (EKI * Eigen::Vector3d(qp1.x, qp1.y, qp1.z)),
            qpt2 = R * (EKI * Eigen::Vector3d(qp2.x, qp2.y, qp2.z));
//      qpt1 /= qpt1[2]; qpt2 /= qpt2[2];
      qpt1.normalize(); qpt2.normalize();
      Qs.reserve(2); Ts.reserve(2);
      Eigen::Quaterniond* rotations = &Qs[0];
      Eigen::Vector3d* translations = &Ts[0];
      int no = theia::TwoPointPosePartialRotation(Yaxis, tpt1, tpt2, qpt1, qpt2, rotations, translations);
      if (no < 2)
      {
         Qs.resize(no);
         Ts.resize(no);
         if (no == 0) return;
      }

      Eigen::Quaterniond QI = Q.inverse();
      for (int i = 0; i < Qs.size(); i++)
      {
         Qs[i] = QI * Qs[i];
         Ts[i] = QI * Ts[i];
      }
#if !defined(NDEBUG)
      for (int i = 0; i < Qs.size(); i++)
      {
         Eigen::Vector3d rpt = Qs[0].toRotationMatrix() * tpt1 + Ts[0];
         rpt /= rpt[2];
         Eigen::Vector3d rrpt(qp1.x, qp1.y, qp1.z);
         rrpt = EKI * rrpt;
         rrpt /= rrpt[2];
         Eigen::Vector3d res = rpt - rrpt;
         std::cout << "TwoPointPosePartialRotation Solution " << (i+1) << ": " << train_pts[i] << " -> " << image_pts[i] << " residual ||" << res.transpose()
                   << "|| = " << res.squaredNorm() << std::endl;
      }
#endif
   }

   double TwoPointPosePartialRotationRANSAC(const cv::Mat& intrinsics, const std::vector<cv::Point3d>& world_pts,
                                          const std::vector<cv::Point3d>& image_pts,
                                          const cv::Vec3f query_g, const cv::Vec3f down_axis,
                                          std::vector<Eigen::Quaterniond>& Qs, std::vector<Eigen::Vector3d>& Ts,
                                          void *RANSAC_params)
//-------------------------------------------------------------------------------------------------------
   {
      if (RANSAC_params == nullptr) throw std::logic_error("poseother::TwoPointPosePartialRotationRANSAC (with depth): "
                                                           "RANSAC params are null");
      Qs.clear(); Ts.clear();
      Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> EK((double *) intrinsics.data);
      Eigen::Matrix3d K = EK;
      Eigen::Matrix3d KI = K.inverse();
      Eigen::Vector3d Yaxis(down_axis[0], down_axis[1], down_axis[2]);
      Eigen::Vector3d query_gravity(query_g[0], query_g[1], query_g[2]);
      Eigen::Quaterniond Q = Eigen::Quaterniond::FromTwoVectors(query_gravity.normalized(), Yaxis);
      Eigen::Quaterniond QI = Q.inverse();
      double confidence;
#ifdef USE_THEIA_RANSAC
      TheiaGravRansacEstimator estimator(K, KI, QI, Yaxis);
      theia::RansacParameters* parameters = static_cast<theia::RansacParameters*>(RANSAC_params);
      theia::RansacSummary summary;
      std::unique_ptr<theia::SampleConsensusEstimator<TheiaGravRansacEstimator>> ransac =
            theia::CreateAndInitializeRansacVariant(theia::RansacType::RANSAC, *parameters, estimator);
      if (ransac)
      {
         std::vector<std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d>> matches;
         for (size_t i=0; i<std::min(world_pts.size(), image_pts.size()); i++)
         {
            const cv::Point3d& tp  = world_pts[i];
            const cv::Point3d& qp  = image_pts[i];
            matches.emplace_back(std::make_tuple(Eigen::Vector3d(tp.x, tp.y, tp.z), Eigen::Vector3d(qp.x, qp.y, qp.z),
                                                 Q*(KI*Eigen::Vector3d(qp.x, qp.y, qp.z))));
         }

         TheiaGravRansacModel best_model;
         if (ransac->Estimate(matches, &best_model, &summary))
         {
            confidence = summary.confidence;
            if (confidence > 0)
            {
               Qs.push_back(best_model.rotation);
               Ts.push_back(best_model.translation);
            }
         }
      }
#else
      TheiaGravRansacData data(KI, Q, world_pts, image_pts);
      templransac::RANSACParams* parameters = static_cast<templransac::RANSACParams*>(RANSAC_params);
      TheiaGravRansacEstimator estimator(KI, Q, QI, Yaxis);
      std::vector<std::pair<double, TheiaGravRansacModel>> results;
      std::vector<std::vector<size_t>> inlier_indices;
      std::stringstream errs;
      Qs.clear();
      Ts.clear();
      confidence = templransac::RANSAC(*parameters, estimator, data, image_pts.size(), 2, 1,
                                       results, inlier_indices, &errs);
      if (results.size() > 0)
      {
         const TheiaGravRansacModel& m = results[0].second;
         Qs.push_back(m.rotation);
         Ts.push_back(m.translation);
         std::cout << "Error " <<  m.error << " ";
      }
#endif
      return confidence;
   }
}

#ifdef USE_THEIA_RANSAC
bool TheiaGravRansacEstimator::EstimateModel(const std::vector<std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d>> &matches,
                                             std::vector<TheiaGravRansacModel> *models) const
//-------------------------------------------------------------------------------------------------------------------------------------
{
   if (matches.size() != 2) return false;
   const Eigen::Vector3d& tpt1 = std::get<0>(matches[0]);
   const Eigen::Vector3d& qpt1 = std::get<2>(matches[0]);
   const Eigen::Vector3d& tpt2 = std::get<0>(matches[1]);
   const Eigen::Vector3d& qpt2 = std::get<2>(matches[1]);
   Eigen::Quaterniond rotations[2];
   Eigen::Vector3d translations[2];
   int no = theia::TwoPointPosePartialRotation(axis, tpt1, tpt2, qpt1, qpt2, rotations, translations);
   for (int i=0; i<no; i++)
   {
      Eigen::Quaterniond Q = QI * rotations[i];
      Eigen::Vector3d T = QI * translations[i];
      models->emplace_back(Q, T);
   }
   return (no > 0);
}

double TheiaGravRansacEstimator::Error(const std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d>& match,
                                       const TheiaGravRansacModel& model) const
//----------------------------------------------------------------------------------------------------------------
{
   Eigen::Vector3d tpt = std::get<0>(match);
   Eigen::Vector3d ipt = model.rotation.toRotationMatrix()*tpt + model.translation;
   ipt /= ipt[2];
   Eigen::Vector3d qpt = KI*std::get<1>(match);
   qpt /= qpt[2];
   Eigen::Vector3d residual = ipt - qpt;
#if !defined(NDEBUG)
   std::cout << "TwoPointPosePartialRotation RANSAC error " << ": (" << tpt.transpose() << ") -> (" << std::get<1>(match).transpose()
             << ") T =(" << model.translation.transpose() << ") residual ||("
             << residual.transpose() << ")|| = " << residual.squaredNorm() << std::endl;

#endif


   return  residual.squaredNorm();
}
#else
const int TheiaGravRansacEstimator::estimate(const TheiaGravRansacData &samples, const std::vector<size_t> &sampleIndices,
                                             templransac::RANSACParams &parameters,
                                             std::vector<TheiaGravRansacModel> &models) const
//-----------------------------------------------------------------------------------------------------------
{
   std::vector<Eigen::Vector3d> world_pts;
   std::vector<Eigen::Vector3d> image_pts;
   const size_t sample_Size = sampleIndices.size();
   for (size_t i=0; i<sample_Size; i++)
   {
      size_t index = sampleIndices[i];
      world_pts.push_back(samples.train_pts[index]);
      image_pts.push_back(samples.rotated_image_pts[index]);
   }
   Eigen::Vector3d tpt1 = world_pts[0], tpt2 = world_pts[1],
                   qpt1 = image_pts[0], qpt2 = image_pts[1];
   Eigen::Quaterniond rotations[2];
   Eigen::Vector3d translations[2];
   int no = theia::TwoPointPosePartialRotation(axis, tpt1, tpt2, qpt1, qpt2, rotations, translations);
   for (int i=0; i<no; i++)
   {
      Eigen::Quaterniond Q = QI * rotations[i];
      Eigen::Vector3d T = QI * translations[i];
      models.emplace_back(Q, T);
   }
   return no;
}

const void TheiaGravRansacEstimator::error(const TheiaGravRansacData &samples, const std::vector<size_t> &sampleIndices,
                                       TheiaGravRansacModel &model, std::vector<size_t> &inlier_indices,
                                       std::vector<size_t> &outlier_indices, double error_threshold) const
//-----------------------------------------------------------------------------------------------------------
{
   std::vector<Eigen::Vector3d> world_pts;
   std::vector<Eigen::Vector3d> image_pts;
   const size_t no = sampleIndices.size();
   for (size_t i=0; i<no; i++)
   {
      size_t index = sampleIndices[i];
      world_pts.push_back(samples.train_pts[index]);
      image_pts.push_back(samples.image_pts[index]);
   }
   double err = error_threshold*error_threshold;
   const Eigen::Matrix3d R = model.rotation.toRotationMatrix();
   const Eigen::Vector3d T = model.translation;
   for (size_t j = 0; j < no; j++)
   {
      Eigen::Vector3d tpt = world_pts[j];
      Eigen::Vector3d ipt = R*tpt + T;
      ipt /= ipt[2];
      Eigen::Vector3d qpt = KI*image_pts[j];
      qpt /= qpt[2];
      Eigen::Vector3d residual = (ipt - qpt);
      model.error = residual.squaredNorm();
      if (model.error < err)
         inlier_indices.push_back(j);
      else
         outlier_indices.push_back(j);
   }
}
#endif