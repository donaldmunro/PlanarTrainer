#include <limits>
#include <utility>
#include <future>

#ifndef POSE_H_48bd9690_f6db_4a6b_80a0_2be822230a23
#define POSE_H_48bd9690_f6db_4a6b_80a0_2be822230a23

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>

class Pose
//========
{
public:
   Pose() : confidence(0) {}

   virtual const char* pose_name() =0;

   virtual void pose(const std::vector<std::pair<cv::Point3d, cv::Point3d>>& pts, void* pparams = nullptr)
   {
      confidence = 0; // set confidence to 1 upon successful completion
   }

   virtual void pose(const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>& pts, void* pparams = nullptr)
   {
      confidence = 0;  // set confidence to 1 upon successful completion
   }

   virtual double pose_ransac(const std::vector<std::pair<cv::Point3d, cv::Point3d>>& pts, int sample_size,
                              void* ransac_parameters, void* pparams =nullptr)
   { confidence = 0;  return false; }

   virtual double pose_ransac(const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>& pts, int sample_size,
                              void* ransac_parameters, void* pparams =nullptr)
   { confidence = 0; return false; }

   virtual void refine(const std::vector<std::pair<cv::Point3d, cv::Point3d>>& pts, int& iterations,
                       void* ransac_parameters = nullptr, void* pparams =nullptr) {}

   virtual void refine(const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>& pts, int& iterations,
                       void* ransac_parameters = nullptr, void* pparams =nullptr) {}

   virtual size_t result_count() { return (confidence <= 0) ? 0 : Qs.size(); }

   virtual bool result(Eigen::Quaterniond& Q, Eigen::Vector3d& T)
   //------------------------------------------------------------------
   {
      if ( (confidence <= 0) || (Qs.size() == 0) || (Ts.size() == 0) )
      {
         double nan = Pose::NaN;
         Q = Eigen::Quaterniond(nan, nan, nan, nan);
         T[0] = T[1] = T[2] = nan;
         return false;
      }
      Q = Qs[0];
      T = Ts[0];
      return true;
   }

   virtual bool result(cv::Mat& R, cv::Vec3d& t)
   //-------------------------------------------
   {
      if ( (confidence <= 0) || (Qs.size() == 0) || (Ts.size() == 0) )
      {
         R = cv::Mat();
         t[0] = t[1] = t[2] = Pose::NaN;
         return false;
      }
      cv::eigen2cv(Qs[0].toRotationMatrix(), R);
      Eigen::Vector3d& T = Ts[0];
      t[0] = T[0]; t[1] = T[1]; t[2] = T[2];
      return true;
   }

   virtual bool result(size_t no, Eigen::Quaterniond& Q, Eigen::Vector3d& T)
   //------------------------------------------------------------------
   {
      size_t no1 = no + 1;
      if ( (confidence <= 0) || (Qs.size() < no1) || (Ts.size() < no1) )
      {
         double nan = Pose::NaN;
         Q = Eigen::Quaterniond(nan, nan, nan, nan);
         T[0] = T[1] = T[2] = nan;
         return false;
      }
      Q = Qs[no];
      T = Ts[no];
      return true;
   }

   virtual bool result(size_t no, cv::Mat& R, cv::Vec3d& t)
   //------------------------------------------------------
   {
      size_t no1 = no + 1;
      if ( (confidence <= 0) || (Qs.size() < no1) || (Ts.size() < no1) )
      {
         R = cv::Mat();
         t[0] = t[1] = t[2] = Pose::NaN;
         return false;
      }
      cv::eigen2cv(Qs[no].toRotationMatrix(), R);
      Eigen::Vector3d& T = Ts[no];
      t[0] = T[0]; t[1] = T[1]; t[2] = T[2];
      return true;
   }

   virtual bool results(std::vector<Eigen::Quaterniond> qs, std::vector<Eigen::Vector3d> ts, bool is_move = false)
   //--------------------------------------------------------------------------------------------------------------
   {
      if ( (confidence <= 0) || (Qs.size() == 0) || (Ts.size() == 0) )
         return false;
      qs = (is_move) ? std::move(Qs) : Qs;
      ts = (is_move) ? std::move(Ts) : Ts;
      return true;
   }

   virtual bool results(std::vector<cv::Mat> Rs, std::vector<cv::Vec3d> ts)
   //-----------------------------------------------------------------------
   {
      if ( (confidence <= 0) || (Qs.size() == 0) || (Ts.size() == 0) )
         return false;
      //if (Qs.size() == 1)
      for (size_t i=0; i<Qs.size(); i++)
      { // There should always be translations corresponding to translations as in those case (FivePointRelative) where
        // there is more than 1 rotation per translation the translation is duplicated
         cv::Mat R;
         cv::eigen2cv(Qs[i].toRotationMatrix(), R);
         Rs.emplace_back(R);
         Eigen::Vector3d& T = Ts[i];
         ts.emplace_back(T[0], T[1], T[2]);
      }
// Probably not really worth it for the number of matrices and their size
//      std::future<void> result( std::async(std::launch::async,
//                                          [this, Rs]()
//                                          {
//                                             for (const Eigen::Quaterniond& Q : Qs)
//                                             {
//                                                cv::Mat R;
//                                                cv::eigen2cv(Q.toRotationMatrix(), R);
//                                                Rs.emplace_back(R);
//                                             }
//                                          }));
//      for (const Eigen::Vector3d& T :Ts)
//         ts.emplace_back(T[0], T[1], T[2]);
//      result.get();
      return true;
   }

   virtual double RANSAC_confidence() { return confidence; }

   virtual void reproject(const cv::Mat& img, const std::vector<std::pair<cv::Point3d, cv::Point3d>>& pts,
                          double& maxError, double& meanError, cv::Mat& imgout, void* pparams =nullptr,
                          cv::Scalar query_color = cv::Scalar(255, 255, 0),
                          cv::Scalar projection_color = cv::Scalar(0, 255, 0)) {}

   virtual void reproject(const cv::Mat& img, const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>& pts,
                          double& maxError, double& meanError, cv::Mat& imgout, void* pparams =nullptr,
                          cv::Scalar query_color = cv::Scalar(255, 255, 0),
                          cv::Scalar projection_color = cv::Scalar(0, 255, 0)) {}

   inline static const double NaN = std::numeric_limits<double>::quiet_NaN();

protected:
   std::vector<Eigen::Quaterniond> Qs;
   std::vector<Eigen::Vector3d> Ts;
   double confidence = 0;

   void eigen2ocv(const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>& in_pts,
                  std::vector<std::pair<cv::Point3d, cv::Point3d>>& out_pts)
   {
      for (const std::pair<Eigen::Vector3d, Eigen::Vector3d>& pp : in_pts)
      {
         const Eigen::Vector3d& tpt = pp.first;
         const Eigen::Vector3d& qpt = pp.second;
         out_pts.emplace_back(std::make_pair(cv::Point3d(tpt[0], tpt[1], tpt[2]), cv::Point3d(qpt[0], qpt[1], qpt[2])));
      }
   }

   void eigen2ocv(const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>& in_pts,
                  std::vector<std::pair<cv::Point3d, cv::Point2d>>& out_pts)
   {
      for (const std::pair<Eigen::Vector3d, Eigen::Vector3d>& pp : in_pts)
      {
         const Eigen::Vector3d& tpt = pp.first;
         const Eigen::Vector3d& qpt = pp.second;
         out_pts.emplace_back(std::make_pair(cv::Point3d(tpt[0], tpt[1], tpt[2]), cv::Point2d(qpt[0], qpt[1])));
      }
   }

   void eigen2ocv(const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>& in_pts,
                  std::vector<std::pair<cv::Point2d, cv::Point2d>>& out_pts)
   {
      for (const std::pair<Eigen::Vector3d, Eigen::Vector3d>& pp : in_pts)
      {
         const Eigen::Vector3d& tpt = pp.first;
         const Eigen::Vector3d& qpt = pp.second;
         out_pts.emplace_back(std::make_pair(cv::Point2d(tpt[0], tpt[1]), cv::Point2d(qpt[0], qpt[1])));
      }
   }
};


#endif //POSE_H_48bd9690_f6db_4a6b_80a0_2be822230a23
