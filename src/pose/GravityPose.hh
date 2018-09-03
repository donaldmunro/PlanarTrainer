#ifndef GRAVITYPOSE_H_9FAEEBA3_3003_41DC_839F_B25FC681F42E
#define GRAVITYPOSE_H_9FAEEBA3_3003_41DC_839F_B25FC681F42E

#include <cmath>

#include "Pose.hh"
#include "pose2d.h"
#include "pose3d.h"
#include "ocv.h"

class Gravity2Gravity2DPose : public Pose
//=====================================
{
public:
   explicit Gravity2Gravity2DPose() : K(cv::Mat()), train_g(NaN, NaN, NaN), query_g(NaN, NaN, NaN)
   {
      Qs.emplace_back(NaN, NaN, NaN, NaN); Ts.emplace_back(NaN, NaN, NaN);
   }
   Gravity2Gravity2DPose(cv::Mat& intrinsics) : K(intrinsics), train_g(NaN, NaN, NaN), query_g(NaN, NaN, NaN)
   {
      Qs.emplace_back(NaN, NaN, NaN, NaN); Ts.emplace_back(NaN, NaN, NaN);
   }

   Gravity2Gravity2DPose(cv::Mat& intrinsics, const Eigen::Vector3d& train_gravity, const Eigen::Vector3d& query_gravity) :
      K(intrinsics), train_g(train_gravity), query_g(query_gravity)
   {
      Qs.emplace_back(NaN, NaN, NaN, NaN); Ts.emplace_back(NaN, NaN, NaN);
   }

   const char* pose_name() override { return name; }

   void pose(const std::vector<std::pair<cv::Point3d, cv::Point3d>>& pts, void* pparams) override
   //---------------------------------------------------------------------------------------------------------------------
   {
      confidence = 0;
      if (pparams != nullptr)
      {
         PoseParameters* params = static_cast<Gravity2Gravity2DPose::PoseParameters*>(pparams);
         pose2d::pose(pts, params->train_gravity, params->query_gravity, (params->K) ? *params->K : K,
                      Qs[0], Ts[0]);
      }
      else
         pose2d::pose(pts, train_g, query_g, K, Qs[0], Ts[0]);
      confidence = 1;
   }

   virtual void pose(const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>& pts, void* pparams) override
   //--------------------------------------------------------------------------------------------------------------
   {
      confidence = 0;
      std::vector<std::pair<cv::Point3d, cv::Point3d>> ppts;
      eigen2ocv(pts, ppts);
      if (pparams != nullptr)
      {
         PoseParameters* params = static_cast<Gravity2Gravity2DPose::PoseParameters*>(pparams);
         pose2d::pose(ppts, params->train_gravity, params->query_gravity, (params->K) ? *params->K : K,
                      Qs[0], Ts[0]);
      }
      else
         pose2d::pose(ppts, train_g, query_g, K, Qs[0], Ts[0]);
      confidence = 1;
   }

   virtual double pose_ransac(const std::vector<std::pair<cv::Point3d, cv::Point3d>>& pts, int sample_size,
                              void* ransac_parameters, void* pparams) override
   //---------------------------------------------------------------------------------------------------------------
   {
      if (sample_size < 0) sample_size = 3;
      if (pparams != nullptr)
      {
         PoseParameters* params = static_cast<Gravity2Gravity2DPose::PoseParameters*>(pparams);
         confidence = pose2d::pose_ransac(pts, params->train_gravity, params->query_gravity,
                                          (params->K) ? *params->K : K, Qs[0], Ts[0], ransac_parameters, sample_size);
      }
      else
         confidence = pose2d::pose_ransac(pts, train_g, query_g, K, Qs[0], Ts[0], ransac_parameters, sample_size);
      return confidence;
   }

   virtual double pose_ransac(const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>& pts, int sample_size,
                              void* ransac_parameters, void* pparams) override
   //---------------------------------------------------------------------------------------------------------------------
   {
      if (sample_size < 0) sample_size = 3;
      std::vector<std::pair<cv::Point3d, cv::Point3d>> ppts;
      eigen2ocv(pts, ppts);
      if (pparams != nullptr)
      {
         PoseParameters* params = static_cast<Gravity2Gravity2DPose::PoseParameters*>(pparams);
         confidence = pose2d::pose_ransac(ppts, params->train_gravity, params->query_gravity,
                                          (params->K) ? *params->K : K, Qs[0], Ts[0], ransac_parameters, sample_size);
      }
      else
         confidence = pose2d::pose_ransac(ppts, train_g, query_g, K, Qs[0], Ts[0], ransac_parameters, sample_size);
      return confidence;
   }

   struct PoseParameters
   {
      cv::Mat* K = nullptr;
      Eigen::Vector3d train_gravity{NaN, NaN, NaN}, query_gravity{NaN, NaN, NaN};
      double depth = 0;
   };
   
protected:
   cv::Mat K;
   Eigen::Vector3d train_g, query_g;

private:
   inline const static char *name = static_cast<const char *>("Gravity2Gravity2DPose");
};

class Gravity2Gravity2DDepthPose : public Gravity2Gravity2DPose
//=============================================================
{
public:
   explicit Gravity2Gravity2DDepthPose() : Gravity2Gravity2DPose(), depth(NaN) {}

   Gravity2Gravity2DDepthPose(cv::Mat& intrinsics) : Gravity2Gravity2DPose(intrinsics), depth(NaN) {}

   Gravity2Gravity2DDepthPose(cv::Mat& intrinsics, const Eigen::Vector3d& train_gravity, const Eigen::Vector3d& query_gravity, double depth) :
         Gravity2Gravity2DPose(intrinsics, train_gravity, query_gravity), depth(depth) { }

   const char* pose_name() override { return name; }

   void pose(const std::vector<std::pair<cv::Point3d, cv::Point3d>>& pts, void* pparams) override
   //--------------------------------------------------------------------------------------------
   {
      confidence = 0;
      if (pparams != nullptr)
      {
         PoseParameters* params = static_cast<Gravity2Gravity2DPose::PoseParameters*>(pparams);
         pose2d::pose(pts, params->train_gravity, params->query_gravity, params->depth, (params->K) ? *params->K : K,
                      Qs[0], Ts[0]);
      }
      else
         pose2d::pose(pts, train_g, query_g, depth, K, Qs[0], Ts[0]);
      confidence = 1;
   }

   void pose(const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>& pts, void* pparams) override
   //--------------------------------------------------------------------------------------------------------------
   {
      confidence = 0;
      std::vector<std::pair<cv::Point3d, cv::Point3d>> ppts;
      eigen2ocv(pts, ppts);
      if (pparams != nullptr)
      {
         PoseParameters* params = static_cast<Gravity2Gravity2DPose::PoseParameters*>(pparams);
         pose2d::pose(ppts, params->train_gravity, params->query_gravity, params->depth, (params->K) ? *params->K : K,
                      Qs[0], Ts[0]);
      }
      else
         pose2d::pose(ppts, train_g, query_g, depth, K, Qs[0], Ts[0]);
      confidence = 1;
   }

   double pose_ransac(const std::vector<std::pair<cv::Point3d, cv::Point3d>>& pts, int sample_size,
                      void* ransac_parameters, void* pparams) override
   //------------------------------------------------------------------------------------------------------
   {
      if (sample_size < 0) sample_size = 3;
      if (pparams != nullptr)
      {
         PoseParameters* params = static_cast<Gravity2Gravity2DPose::PoseParameters*>(pparams);
         confidence = pose2d::pose_ransac(pts, params->train_gravity, params->query_gravity, params->depth,
                                          (params->K) ? *params->K : K, Qs[0], Ts[0], ransac_parameters, sample_size);
      }
      else
         confidence = pose2d::pose_ransac(pts, train_g, query_g, depth, K, Qs[0], Ts[0], ransac_parameters, sample_size);
      return confidence;
   }

   virtual double pose_ransac(const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>& pts, int sample_size,
                              void* ransac_parameters, void* pparams) override
   //---------------------------------------------------------------------------------------------------------------------
   {
      if (sample_size < 0) sample_size = 3;
      std::vector<std::pair<cv::Point3d, cv::Point3d>> ppts;
      eigen2ocv(pts, ppts);
      if (pparams != nullptr)
      {
         PoseParameters* params = static_cast<Gravity2Gravity2DPose::PoseParameters*>(pparams);
         confidence = pose2d::pose_ransac(ppts, params->train_gravity, params->query_gravity, params->depth,
                                          (params->K) ? *params->K : K, Qs[0], Ts[0], ransac_parameters, sample_size);
      }
      else
         confidence = pose2d::pose_ransac(ppts, train_g, query_g, depth, K, Qs[0], Ts[0], ransac_parameters, sample_size);
      return confidence;
   }

   void reproject(const cv::Mat& img, const std::vector<std::pair<cv::Point3d, cv::Point3d>>& pts,
                  double& maxError, double& meanError, cv::Mat& imgout, void* pparams =nullptr,
                  cv::Scalar query_color = cv::Scalar(255, 255, 0),
                  cv::Scalar projection_color = cv::Scalar(0, 255, 0)) override
   //----------------------------------------------------------------------------------------------
   {
      Eigen::Matrix3d R = Qs[0].toRotationMatrix();
      Eigen::Vector3d T = Ts[0];
      Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> EK((double*) K.data);
      std::vector<cv::Point3d> tpts, qpts;
      for (const std::pair<cv::Point3d, cv::Point3d>& pp : pts)
      {
         tpts.emplace_back(pp.first);
         qpts.emplace_back(pp.second);
      }
      if (pparams != nullptr)
      {
         PoseParameters* params = static_cast<Gravity2Gravity2DPose::PoseParameters*>(pparams);
         Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> EK((double*) params->K->data);
         ocv::reprojection2d(EK, R, T, tpts, qpts, params->depth, maxError, meanError, &img, &imgout, true, query_color, projection_color);
      }
      else
      {
         Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> EK((double*) K.data);
         ocv::reprojection2d(EK, R, T, tpts, qpts, depth, maxError, meanError, &img, &imgout, true, query_color, projection_color);
      }
   }

   void reproject(const cv::Mat& img, const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>& pts,
                  double& maxError, double& meanError, cv::Mat& imgout, void* pparams =nullptr,
                  cv::Scalar query_color = cv::Scalar(255, 255, 0),
                  cv::Scalar projection_color = cv::Scalar(0, 255, 0)) override
   //----------------------------------------------------------------------------------------------------------------------
   {
      Eigen::Matrix3d R = Qs[0].toRotationMatrix();
      Eigen::Vector3d T = Ts[0];
      Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> EK((double*) K.data);
      std::vector<cv::Point3d> tpts, qpts;
      for (const std::pair<Eigen::Vector3d, Eigen::Vector3d>& pp : pts)
      {
         const Eigen::Vector3d& tpt = pp.first;
         const Eigen::Vector3d& qpt = pp.second;
         tpts.emplace_back(tpt[0], tpt[1], tpt[2]);
         qpts.emplace_back(qpt[0], qpt[1], qpt[2]);
      }
      if (pparams != nullptr)
      {
         PoseParameters* params = static_cast<Gravity2Gravity2DPose::PoseParameters*>(pparams);
         Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> EK((double*) params->K->data);
         ocv::reprojection2d(EK, R, T, tpts, qpts, params->depth, maxError, meanError, &img, &imgout, true, query_color, projection_color);
      }
      else
      {
         Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> EK((double*) K.data);
         ocv::reprojection2d(EK, R, T, tpts, qpts, depth, maxError, meanError, &img, &imgout, true, query_color, projection_color);
      }
   }

protected:
   double depth;

private:
   inline const static char *name = static_cast<const char *>("Gravity2Gravity2DDepthPose");
};

class Gravity2Gravity3DPose : public Pose
//=====================================
{
public:
   explicit Gravity2Gravity3DPose() : KI(), train_g(NaN, NaN, NaN), query_g(NaN, NaN, NaN)
   {
      Qs.emplace_back(NaN, NaN, NaN, NaN);
      Ts.emplace_back(NaN, NaN, NaN);
   }

   Gravity2Gravity3DPose(cv::Mat& intrinsics) : train_g(NaN, NaN, NaN), query_g(NaN, NaN, NaN)
   //-----------------------------------------------------------------------------------------
   {
      Qs.emplace_back(NaN, NaN, NaN, NaN); Ts.emplace_back(NaN, NaN, NaN);
      Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> K((double*) intrinsics.data);
      KI = K.inverse();
   }

   Gravity2Gravity3DPose(cv::Mat& intrinsics, const Eigen::Vector3d train_gravity, const Eigen::Vector3d& query_gravity) :
         train_g(train_gravity), query_g(query_gravity)
   //--------------------------------------------------------------------------------------------------------------------
   {
      Qs.emplace_back(NaN, NaN, NaN, NaN); Ts.emplace_back(NaN, NaN, NaN);
      Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> K((double*) intrinsics.data);
      KI = K.inverse();
   }

   const char* pose_name() override { return name; }

   void pose(const std::vector<std::pair<cv::Point3d, cv::Point3d>>& pts, void* pparams) override
   //---------------------------------------------------------------------------------------------
   {
      confidence = 0;
      std::vector<std::pair<cv::Point3d, cv::Point2d>> ppts;
      for (const std::pair<cv::Point3d, cv::Point3d>& pp : pts)
      {
         const cv::Point3d &wpt = pp.first, &ipt = pp.second;
         ppts.emplace_back(std::make_pair(wpt, cv::Point2d(ipt.x, ipt.y)));
      }
      if (pparams != nullptr)
      {
         PoseParameters* params = static_cast<Gravity2Gravity3DPose::PoseParameters*>(pparams);
         if (params->intrinsics != nullptr)
         {
            Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> K((double*) params->intrinsics->data);
            pose3d::pose(ppts, params->train_gravity, params->query_gravity, K.inverse(), Qs[0], Ts[0]);
         }
         else
            pose3d::pose(ppts, params->train_gravity, params->query_gravity, (params->pKI) ? *params->pKI :KI,
                         Qs[0], Ts[0]);
      }
      else
         pose3d::pose(ppts, train_g, query_g, KI, Qs[0], Ts[0]);
      confidence = 1;
   }

   void pose(const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>& pts, void* pparams) override
   //-----------------------------------------------------------------------------------------------------
   {
      confidence = 0;
      std::vector<std::pair<cv::Point3d, cv::Point2d>> ppts;
      eigen2ocv(pts, ppts);
      if (pparams != nullptr)
      {
         PoseParameters* params = static_cast<Gravity2Gravity3DPose::PoseParameters*>(pparams);
         if (params->intrinsics != nullptr)
         {
            Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> K((double*) params->intrinsics->data);
            pose3d::pose(ppts, params->train_gravity, params->query_gravity, K.inverse(), Qs[0], Ts[0]);
         }
         else
            pose3d::pose(ppts, params->train_gravity, params->query_gravity, (params->pKI) ? *params->pKI :KI,
                         Qs[0], Ts[0]);
      }
      else
         pose3d::pose(ppts, train_g, query_g, KI, Qs[0], Ts[0]);
      confidence = 1;
   }

   double pose_ransac(const std::vector<std::pair<cv::Point3d, cv::Point3d>>& pts, int sample_size,
                      void* ransac_parameters, void* pparams) override
   //------------------------------------------------------------------------------------------------------
   {
      if (sample_size < 0) sample_size = 3;
      std::vector<std::pair<cv::Point3d, cv::Point2d>> ppts;
      for (const std::pair<cv::Point3d, cv::Point3d>& pp : pts)
      {
         const cv::Point3d &wpt = pp.first, &ipt = pp.second;
         ppts.emplace_back(std::make_pair(wpt, cv::Point2d(ipt.x, ipt.y)));
      }
      if (pparams != nullptr)
      {
         PoseParameters* params = static_cast<Gravity2Gravity3DPose::PoseParameters*>(pparams);
         if (params->intrinsics != nullptr)
         {
            Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> K((double*) params->intrinsics->data);
            confidence = pose3d::pose_ransac(ppts, params->train_gravity, params->query_gravity, K.inverse(),
                                              Qs[0], Ts[0], ransac_parameters, sample_size);
         }
         else
            confidence = pose3d::pose_ransac(ppts, params->train_gravity, params->query_gravity,
                                             (params->pKI) ? *params->pKI : KI, Qs[0], Ts[0],
                                             ransac_parameters, sample_size);
      }
      else
         confidence = pose3d::pose_ransac(ppts, train_g, query_g, KI, Qs[0], Ts[0], ransac_parameters, sample_size);
      return confidence;
   }

   double pose_ransac(const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>& pts, int sample_size,
                      void* ransac_parameters, void* pparams) override
   //-------------------------------------------------------------------------------------------------------------
   {
      if (sample_size < 0) sample_size = 3;
      std::vector<std::pair<cv::Point3d, cv::Point2d>> ppts;
      eigen2ocv(pts, ppts);
      if (pparams != nullptr)
      {
         PoseParameters* params = static_cast<Gravity2Gravity3DPose::PoseParameters*>(pparams);
         if (params->intrinsics != nullptr)
         {
            Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> K((double*) params->intrinsics->data);
            confidence = pose3d::pose_ransac(ppts, params->train_gravity, params->query_gravity, K.inverse(),
                                             Qs[0], Ts[0], ransac_parameters, sample_size);
         }
         else
            confidence = pose3d::pose_ransac(ppts, params->train_gravity, params->query_gravity,
                                             (params->pKI) ? *params->pKI : KI, Qs[0], Ts[0],
                                             ransac_parameters, sample_size);
      }
      else
         confidence = pose3d::pose_ransac(ppts, train_g, query_g, KI, Qs[0], Ts[0], ransac_parameters, sample_size);
      return confidence;
   }

   void reproject(const cv::Mat& img, const std::vector<std::pair<cv::Point3d, cv::Point3d>>& pts,
                  double& maxError, double& meanError, cv::Mat& imgout, void* pparams =nullptr,
                  cv::Scalar query_color = cv::Scalar(255, 255, 0),
                  cv::Scalar projection_color = cv::Scalar(0, 255, 0)) override
   //----------------------------------------------------------------------------------------------
   {
      Eigen::Matrix3d R = Qs[0].toRotationMatrix();
      Eigen::Vector3d T = Ts[0];
      std::vector<cv::Point3d> wpts;
      std::vector<cv::Point2d> ipts;
      for (const std::pair<cv::Point3d, cv::Point3d>& pp : pts)
      {
         wpts.emplace_back(pp.first);
         const cv::Point3d& qpt = pp.second;
         ipts.emplace_back(qpt.x, qpt.y);
      }
      if (pparams != nullptr)
      {
         PoseParameters* params = static_cast<Gravity2Gravity3DPose::PoseParameters*>(pparams);
         Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> EK((double*) params->intrinsics->data);
         ocv::reprojection3d(EK, R, T, wpts, ipts, maxError, meanError, &img, &imgout, true, query_color, projection_color);
      }
      else
      {
         Eigen::Matrix3d K = KI.inverse();
         ocv::reprojection3d(K, R, T, wpts, ipts, maxError, meanError, &img, &imgout, true, query_color, projection_color);
      }
   }

   void reproject(const cv::Mat& img, const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>& pts,
                  double& maxError, double& meanError, cv::Mat& imgout, void* pparams =nullptr,
                  cv::Scalar query_color = cv::Scalar(255, 255, 0),
                  cv::Scalar projection_color = cv::Scalar(0, 255, 0)) override
   //----------------------------------------------------------------------------------------------------------------------
   {
      Eigen::Matrix3d R = Qs[0].toRotationMatrix();
      Eigen::Vector3d T = Ts[0];
      std::vector<cv::Point3d> wpts;
      std::vector<cv::Point2d> ipts;
      for (const std::pair<Eigen::Vector3d, Eigen::Vector3d>& pp : pts)
      {
         const Eigen::Vector3d& tpt = pp.first;
         const Eigen::Vector3d& qpt = pp.second;
         wpts.emplace_back(tpt[0], tpt[1], tpt[2]);
         ipts.emplace_back(qpt[0], qpt[1]);
      }
      if (pparams != nullptr)
      {
         PoseParameters* params = static_cast<Gravity2Gravity3DPose::PoseParameters*>(pparams);
         Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> K((double*) params->intrinsics->data);
         ocv::reprojection3d(K, R, T, wpts, ipts, maxError, meanError, &img, &imgout, true, query_color, projection_color);
      }
      else
      {
         Eigen::Matrix3d K = KI.inverse();
         ocv::reprojection3d(K, R, T, wpts, ipts, maxError, meanError, &img, &imgout, true, query_color, projection_color);
      }
   }

   struct PoseParameters
   {
      cv::Mat* intrinsics = nullptr;
      Eigen::Matrix3d* pKI = nullptr;
      Eigen::Vector3d train_gravity{NaN, NaN, NaN}, query_gravity{NaN, NaN, NaN};
   };

protected:
   Eigen::Matrix3d KI;
   Eigen::Vector3d train_g, query_g;

private:
   inline const static char *name = static_cast<const char *>("Gravity2Gravity3DPose");
};

#endif //TRAINER_GRAVITYPOSE_H
