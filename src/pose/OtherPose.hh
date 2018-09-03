#ifndef OTHERPOSE_HH_EC1B3022_CE6A_40AF_8192_E0E914EF25D6
#define OTHERPOSE_HH_EC1B3022_CE6A_40AF_8192_E0E914EF25D6

#include "Pose.hh"
#include "poseocv.h"
#include <ocv.h>

class PnP3DPose : public Pose
//=====================================
{
public:
   PnP3DPose() {}
   PnP3DPose(cv::Mat& intrinsics, int flags = cv::SOLVEPNP_ITERATIVE) : K(intrinsics), flags(flags) { }

   const char* pose_name() override { return name; }

   void pose(const std::vector<std::pair<cv::Point3d, cv::Point3d>>& pts, void* pparams) override
   //--------------------------------------------------------------------------------------------
   {
      std::vector<cv::Point3d> world_pts, query_pts;
      for (const std::pair<cv::Point3d, cv::Point3d>& pp : pts)
      {
         const cv::Point3d &wpt = pp.first, &ipt = pp.second;
         world_pts.emplace_back(wpt.x, wpt.y, wpt.z);
         query_pts.emplace_back(ipt.x, ipt.y, ipt.z);
      }
      if (pparams != nullptr)
      {
         PoseParameters* params = static_cast<PoseParameters *>(pparams);
         PnP_pose((params->K != nullptr) ? *params->K : K, world_pts, query_pts, false, params->flags, nullptr);
      }
      else
         PnP_pose(K, world_pts, query_pts, false, flags, nullptr);
   }

   void pose(const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>& pts, void* pparams) override
   //----------------------------------------------------------------------------------------------------
   {
      std::vector<cv::Point3d> world_pts, query_pts;
      for (const std::pair<Eigen::Vector3d, Eigen::Vector3d>& pp : pts)
      {
         const Eigen::Vector3d &wpt = pp.first, &ipt = pp.second;
         world_pts.emplace_back(wpt[0], wpt[1], wpt[2]);
         query_pts.emplace_back(ipt[0], ipt[1], ipt[2]);
      }
      if (pparams != nullptr)
      {
         PoseParameters* params = static_cast<PoseParameters *>(pparams);
         PnP_pose((params->K != nullptr) ? *params->K : K, world_pts, query_pts, false, params->flags, nullptr);
      }
      else
         PnP_pose(K, world_pts, query_pts, false, flags, nullptr);
   }

   double pose_ransac(const std::vector<std::pair<cv::Point3d, cv::Point3d>>& pts, int sample_size,
                      void* ransac_parameters, void* pparams) override
   //----------------------------------------------------------------------------------------------------------------
   {
      std::vector<cv::Point3d> world_pts, query_pts;
      for (const std::pair<cv::Point3d, cv::Point3d>& pp : pts)
      {
         const cv::Point3d &wpt = pp.first, &ipt = pp.second;
         world_pts.emplace_back(wpt.x, wpt.y, wpt.z);
         query_pts.emplace_back(ipt.x, ipt.y, ipt.z);
      }
      poseocv::PnPRANSACParameters* pRANSACParams = static_cast<poseocv::PnPRANSACParameters*>(ransac_parameters);
      if (pparams != nullptr)
      {
         PoseParameters* params = static_cast<PoseParameters *>(pparams);
         PnP_pose((params->K != nullptr) ? *params->K : K, world_pts, query_pts, false, params->flags, pRANSACParams);
      }
      else
         PnP_pose(K, world_pts, query_pts, false, flags, pRANSACParams);
      return confidence;
   }

   double pose_ransac(const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>& pts, int sample_size,
                      void* ransac_parameters, void* pparams) override
   //---------------------------------------------------------------------------------------------------------------------
   {
      poseocv::PnPRANSACParameters* pRANSACParams = static_cast<poseocv::PnPRANSACParameters*>(ransac_parameters);
      std::vector<cv::Point3d> world_pts, query_pts;
      for (const std::pair<Eigen::Vector3d, Eigen::Vector3d>& pp : pts)
      {
         const Eigen::Vector3d &wpt = pp.first, &ipt = pp.second;
         world_pts.emplace_back(wpt[0], wpt[1], wpt[2]);
         query_pts.emplace_back(ipt[0], ipt[1], ipt[2]);
      }
      if (pparams != nullptr)
      {
         PoseParameters* params = static_cast<PoseParameters *>(pparams);
         PnP_pose((params->K != nullptr) ? *params->K : K, world_pts, query_pts, false, params->flags, pRANSACParams);
      }
      else
         PnP_pose(K, world_pts, query_pts, false, flags, pRANSACParams);
      return confidence;
   }

   void refine(const std::vector<std::pair<cv::Point3d, cv::Point3d>>& pts, int& iterations,
               void* ransac_parameters, void* pparams) override
   //-----------------------------------------------------------------------------------------------------
   {
      if ( (confidence <= 0) || (Qs.empty()) || (Ts.empty()) || (RV.empty()) )
      {
         iterations = -1;
         return;
      }
      poseocv::PnPRANSACParameters* pRANSACParams = static_cast<poseocv::PnPRANSACParameters*>(ransac_parameters);
      std::vector<cv::Point3d> world_pts, query_pts;
      for (const std::pair<cv::Point3d, cv::Point3d>& pp : pts)
      {
         const cv::Point3d &wpt = pp.first, &ipt = pp.second;
         world_pts.emplace_back(wpt.x, wpt.y, wpt.z);
         query_pts.emplace_back(ipt.x, ipt.y, ipt.z);
      }
      if (pparams != nullptr)
      {
         PoseParameters* params = static_cast<PoseParameters *>(pparams);
         PnP_pose((params->K != nullptr) ? *params->K : K, world_pts, query_pts, true, params->flags, pRANSACParams);
      }
      else
         PnP_pose(K, world_pts, query_pts, true, flags, pRANSACParams);
   }

   void refine(const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>& pts, int& iterations,
               void* ransac_parameters, void* pparams) override
   //-------------------------------------------------------------------------------------------------------
   {
      if ( (confidence <= 0) || (Qs.empty()) || (Ts.empty()) || (RV.empty()) )
      {
         iterations = -1;
         return;
      }
      poseocv::PnPRANSACParameters* pRANSACParams = static_cast<poseocv::PnPRANSACParameters*>(ransac_parameters);
      std::vector<cv::Point3d> world_pts, query_pts;
      for (const std::pair<Eigen::Vector3d, Eigen::Vector3d>& pp : pts)
      {
         const Eigen::Vector3d &wpt = pp.first, &ipt = pp.second;
         world_pts.emplace_back(wpt[0], wpt[1], wpt[2]);
         query_pts.emplace_back(ipt[0], ipt[1], ipt[2]);
      }
      if (pparams != nullptr)
      {
         PoseParameters* params = static_cast<PoseParameters *>(pparams);
         PnP_pose((params->K != nullptr) ? *params->K : K, world_pts, query_pts, true, params->flags, pRANSACParams);
      }
      else
         PnP_pose(K, world_pts, query_pts, true, flags, pRANSACParams);
   }

   void reproject(const cv::Mat& img, const std::vector<std::pair<cv::Point3d, cv::Point3d>>& pts,
                  double& maxError, double& meanError, cv::Mat& imgout, void* pparams =nullptr,
                  cv::Scalar query_color = cv::Scalar(255, 255, 0),
                  cv::Scalar projection_color = cv::Scalar(0, 255, 0)) override
   //----------------------------------------------------------------------------------------------
   {
      if ( (Qs.empty()) || (Ts.empty()) ) return;
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
      Eigen::Matrix3d EK;
      if (pparams != nullptr)
      {
         PoseParameters* params = static_cast<PoseParameters*>(pparams);
         if (params->K != nullptr)
         {
            Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> KM((double*) params->K->data);
            EK = KM;
         }
         else
         {
            Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> KM((double*) K.data);
            EK = KM;
         }
         ocv::reprojection3d(EK, R, T, wpts, ipts, maxError, meanError, &img, &imgout, true, query_color, projection_color);
      }
      else
      {
         Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> KM((double*) K.data);
         EK = KM;
         ocv::reprojection3d(EK, R, T, wpts, ipts, maxError, meanError, &img, &imgout, true, query_color, projection_color);
      }
   }

   void reproject(const cv::Mat& img, const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>& pts,
                  double& maxError, double& meanError, cv::Mat& imgout, void* pparams =nullptr,
                  cv::Scalar query_color = cv::Scalar(255, 255, 0),
                  cv::Scalar projection_color = cv::Scalar(0, 255, 0)) override
   //----------------------------------------------------------------------------------------------------------------------
   {
      if ( (Qs.empty()) || (Ts.empty()) ) return;
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
      Eigen::Matrix3d EK;
      if (pparams != nullptr)
      {
         PoseParameters* params = static_cast<PoseParameters*>(pparams);
         if (params->K != nullptr)
         {
            Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> KM((double*) params->K->data);
            EK = KM;
         }
         else
         {
            Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> KM((double*) K.data);
            EK = KM;
         }
         ocv::reprojection3d(EK, R, T, wpts, ipts, maxError, meanError, &img, &imgout, true, query_color, projection_color);
      }
      else
      {
         Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> KM((double*) K.data);
         EK = KM;
         ocv::reprojection3d(EK, R, T, wpts, ipts, maxError, meanError, &img, &imgout, true, query_color, projection_color);
      }
   }

   struct PoseParameters
   {
      cv::Mat* K = nullptr;
      int flags = 0;
   };

protected:
   cv::Mat K, RV, OR, OT;
   int flags;

private:
   void PnP_pose(const cv::Mat K, const std::vector<cv::Point3d>& train_pts, const std::vector<cv::Point3d>& query_pts,
                 bool is_refine, int flags, poseocv::PnPRANSACParameters* pRANSACParams)
   //-----------------------------------------------------------------------------------------------------
   {
      if (! is_refine) confidence = 0;
      Qs.clear(); Ts.clear();
      long time;
      if ( (! poseocv::pose_PnP(train_pts, query_pts, K, RV, OT, &OR, time, is_refine, flags,
                                pRANSACParams)) || (OR.empty()) || (OT.empty()) )
      {
         RV.release();
         return;
      }

      Eigen::Matrix3d ER;
      cv::cv2eigen(OR, ER);
      Qs.emplace_back(ER);
      Eigen::Vector3d T;
      cv::cv2eigen(OT, T);
      Ts.push_back(T);
      confidence = 1;
   }

   inline const static char *name = static_cast<const char *>("PnP3DPose");
};

class Homography2DPose : public Pose
//=====================================
{
public:
   Homography2DPose(cv::Mat& intrinsics) : K(intrinsics) { }

   const char* pose_name() override { return name; }

   void pose(const std::vector<std::pair<cv::Point3d, cv::Point3d>>& pts, void* pparams) override
   //-----------------------------------------------------------------------------------------------------
   {
      is_ransac = false;
      std::vector<cv::Point3d> train_pts, query_pts;
      for (const std::pair<cv::Point3d, cv::Point3d>& pp : pts)
      {
         const cv::Point3d &wpt = pp.first, &ipt = pp.second;
         train_pts.emplace_back(wpt.x, wpt.y, wpt.z);
         query_pts.emplace_back(ipt.x, ipt.y, ipt.z);
      }
      homography_pose(train_pts, query_pts);
   }

   void pose(const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>& pts, void* pparams) override
   //--------------------------------------------------------------------------------------------------------------
   {
      is_ransac = false;
      std::vector<cv::Point3d> train_pts, query_pts;
      for (const std::pair<Eigen::Vector3d, Eigen::Vector3d>& pp : pts)
      {
         const Eigen::Vector3d &tpt = pp.first, &ipt = pp.second;
         train_pts.emplace_back(tpt[0], tpt[1], tpt[2]);
         query_pts.emplace_back(ipt[0], ipt[1], ipt[2]);
      }
      homography_pose(train_pts, query_pts);
   }

   double pose_ransac(const std::vector<std::pair<cv::Point3d, cv::Point3d>>& pts, int sample_size,
                      void* ransac_parameters, void* pparams) override
   //----------------------------------------------------------------------------------------------------------------
   {

      is_ransac = true;
      std::vector<cv::Point3d> train_pts, query_pts;
      for (const std::pair<cv::Point3d, cv::Point3d>& pp : pts)
      {
         const cv::Point3d &wpt = pp.first, &ipt = pp.second;
         train_pts.emplace_back(wpt.x, wpt.y, wpt.z);
         query_pts.emplace_back(ipt.x, ipt.y, ipt.z);
      }
      homography_pose(train_pts, query_pts);
      confidence = (Qs.size() > 0) ? 1 : 0;
      return confidence;
   }

   double pose_ransac(const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>& pts, int sample_size,
                      void* ransac_parameters, void* pparams) override
   //--------------------------------------------------------------------------------------------------------------
   {
      is_ransac = true;
      std::vector<cv::Point3d> train_pts, query_pts;
      for (const std::pair<Eigen::Vector3d, Eigen::Vector3d>& pp : pts)
      {
         const Eigen::Vector3d &tpt = pp.first, &ipt = pp.second;
         train_pts.emplace_back(tpt[0], tpt[1], tpt[2]);
         query_pts.emplace_back(ipt[0], ipt[1], ipt[2]);
      }
      homography_pose(train_pts, query_pts);
      confidence = (Qs.size() > 0) ? 1 : 0;
      return confidence;
   }

   std::vector<Eigen::Vector3d> Ns;

protected:
   cv::Mat K;
   Eigen::Vector3d train_g, query_g;


private:
   void homography_pose(const std::vector<cv::Point3d>& train_pts, const std::vector<cv::Point3d>& query_pts)
   //-----------------------------------------------------------------------------------------------------
   {
      confidence = 0;
      std::vector<cv::Mat> rotations, translations, normals;
      poseocv::homography_pose(K, train_pts, query_pts, rotations, translations, normals, is_ransac);
      if (rotations.empty()) return;
      for (size_t solution_no = 0; solution_no < rotations.size(); solution_no++)
      {
         cv::Mat& rotation = rotations[solution_no];
         Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> R((double*) rotation.data);
         Qs.emplace_back(R);
         cv::Mat& translation = translations[solution_no];
         Ts.emplace_back(translation.at<double>(0, 0), translation.at<double>(1, 0), translation.at<double>(2, 0));
         cv::Mat& normal = normals[solution_no];
         Ns.emplace_back(normal.at<double>(0, 0), normal.at<double>(1, 0), normal.at<double>(2, 0));
      }
      confidence = 1;
   }

   bool is_ransac = false;
   inline const static char *name = static_cast<const char *>("Homography2DPose");
};

#endif //TRAINER_OTHERPOSE_HH
