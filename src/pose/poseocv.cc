#include <iostream>
#include <iomanip>
#include <chrono>
#include <limits>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

#include "poseocv.h"
#include "math.hh"
#include "display.h"

namespace poseocv
{
   void homography_pose(const cv::Mat& intrinsics, const std::vector<cv::Point3d>& train_img_pts,
                        const std::vector<cv::Point3d>& query_img_pts, std::vector<cv::Mat>& rotations,
                        std::vector<cv::Mat>& translations, std::vector<cv::Mat>& normals, bool isRANSAC)
//---------------------------------------------------------------------------------------------
   {
      cv::Mat H;
      rotations.clear(); translations.clear();
      std::vector<cv::Point2f> train_pts, query_pts;
      for (size_t i = 0; i < std::min(train_img_pts.size(), query_img_pts.size()); i++)
      {
         const cv::Point3d tpt = train_img_pts[i];
         train_pts.emplace_back(tpt.x, tpt.y);
         const cv::Point3d qpt = query_img_pts[i];
         query_pts.emplace_back(qpt.x, qpt.y);
      }
      if ((isRANSAC) && (train_img_pts.size() > 3))
         H = cv::findHomography(train_pts, query_pts, cv::RANSAC);
      else
         H = cv::findHomography(train_pts, query_pts);
      if (H.empty()) return;
      cv::decomposeHomographyMat(H, intrinsics, rotations, translations, normals);
   }


   bool pose_PnP(const std::vector<cv::Point3d>& world_pts, const std::vector<cv::Point3d>& image_pts,
                 const cv::Mat& intrinsics, cv::Mat& rotation_vec, cv::Mat& translations3x1, cv::Mat* R,
                 long& total_time, bool is_refine, int flags, PnPRANSACParameters* RANSACParams)
//--------------------------------------------------------------------------------------------------
   {
      std::vector<cv::Point2f> points2d;
      std::vector<cv::Point3f> points3d;
      std::size_t n = std::min(world_pts.size(), image_pts.size());
      for (size_t i = 0; i < n; i++)
      {
         if ((i >= 4) &&
             ((flags == cv::SOLVEPNP_P3P) || (flags == cv::SOLVEPNP_AP3P)))
         {
//         std::cout << "poseocv::pose_PnP WARNING: Abbreviated input point count to " << points2d.size() << std::endl;
            break;
         }
         cv::Point3d pt = image_pts[i];
         points2d.emplace_back(pt.x, pt.y);
         pt = world_pts[i];
         points3d.emplace_back(pt.x, pt.y, pt.z);
      }
      if ( (RANSACParams != nullptr) && (n > 4) )
      {
         std::vector<unsigned char> inliers(points3d.size());
         auto start = std::chrono::high_resolution_clock::now();
         if (cv::solvePnPRansac(points3d, points2d, intrinsics, cv::noArray(), rotation_vec, translations3x1,
                                is_refine, RANSACParams->iterationsCount, RANSACParams->reprojectionError,
                                RANSACParams->confidence, inliers, flags))
         {
            auto diff = std::chrono::high_resolution_clock::now() - start;
            total_time = std::chrono::duration_cast<std::chrono::nanoseconds>(diff).count();
//         for (size_t j = 0; j < points3d.size(); j++)
//         {
//            if (!inliers[j])
//               fprintf(stdout, "outlier (%.5f, %.5f) (%.5f, %.5f, %.5f)\n", points2d[j].x, points2d[j].y, points3d[j].x,
//                       points3d[j].y, points3d[j].z);
//         }
            if (R != nullptr)
               cv::Rodrigues(rotation_vec, *R);
            return true;
         }
         else
            total_time = -1;
      }
      else
      {
         auto start = std::chrono::high_resolution_clock::now();
         try
         {
            if (cv::solvePnP(points3d, points2d, intrinsics, cv::noArray(), rotation_vec, translations3x1, is_refine,
                             flags))
            {
               auto diff = std::chrono::high_resolution_clock::now() - start;
               total_time = std::chrono::duration_cast<std::chrono::nanoseconds>(diff).count();
               if (R != nullptr)
                  cv::Rodrigues(rotation_vec, *R);
               return true;
            }
            else
               total_time = -1;
         }
         catch (...)
         {
            return false;
         }
      }
      return false;
   }

   void display_PnP(const std::vector<cv::Point3d>& world_points,
                    const std::vector<cv::Point3d>& image_points,
                    const cv::Mat& intrinsics, bool isMinimal, const cv::Mat* query_img,
                    bool show_reprojection, bool save_reprojection, bool is_time)
//----------------------------------------------------------------------------------------------------
   {
      cv::Mat rotation_vec, translations3x1, R;
      std::array<std::pair<std::string, int>, 5> methods {std::make_pair("Iterative", cv::SOLVEPNP_ITERATIVE),
                                                          std::make_pair("P3P", cv::SOLVEPNP_P3P),
                                                          std::make_pair("EPnP", cv::SOLVEPNP_EPNP),
                                                          std::make_pair("DLS", cv::SOLVEPNP_DLS),
                                                          std::make_pair("AP3P", cv::SOLVEPNP_AP3P)
      };
      for (auto pp : methods)
      {
         bool is_pose;
         std::vector<cv::Point3d> world_pts, image_pts, new_world_pts, new_image_pts;
         if (world_points.size() != image_points.size())
         {
            size_t n = std::min(world_points.size(), image_points.size());
            std::copy(world_points.begin(), world_points.begin() + n, std::back_inserter(world_pts));
            std::copy(image_points.begin(), image_points.begin() + n, std::back_inserter(image_pts));
         }
         else
         {
            world_pts = world_points;
            image_pts = image_points;
         }
         switch (pp.second)
         {
            case cv::SOLVEPNP_AP3P:
            case cv::SOLVEPNP_P3P:
               if (world_pts.size() > 4)
               {
                  std::copy(world_pts.begin(), world_pts.begin() + 4, std::back_inserter(new_world_pts));
                  std::copy(image_pts.begin(), image_pts.begin() + 4, std::back_inserter(new_image_pts));
               }
         }
         long time_ns;
         for (int i = 0; i < 1; i++)
         {
            bool is_ransac = (i > 0);
            PnPRANSACParameters RANSAC_params, *pRANSAC_params = nullptr;
            if (is_ransac)
            {
               std::cout << "RANSAC:";
               RANSAC_params.iterationsCount = 1000;
               pRANSAC_params = &RANSAC_params;
            }
            try
            {
               if (new_world_pts.size() > 0)
                  is_pose = pose_PnP(new_world_pts, new_image_pts, intrinsics, rotation_vec, translations3x1, &R,
                                     time_ns, false, pp.second, pRANSAC_params);
               else
                  is_pose = pose_PnP(world_pts, image_pts, intrinsics, rotation_vec, translations3x1, &R,
                                     time_ns, false, pp.second, pRANSAC_params);
            }
            catch (std::exception& e)
            {
//            std::cerr << "   display_PnP " << pp.first << " Exception: " << e.what() << std::endl;
               std::cout << "    No PnP solution (exception)" << std::endl;
               continue;
            }
            if (is_pose)
            {
               std::cout << "    ***** " << pp.first << " (" << time_ns << "ns )" << std::endl;
               cv::Mat TT = translations3x1.t();
               Eigen::Matrix3d ER;
               cv::cv2eigen(R, ER);
               Eigen::AngleAxisd aa(ER);
               Eigen::Quaterniond QQ(ER);
               if (!isMinimal)
               {
                  std::cout << "    Axis: " << aa.axis().transpose() << " Angle: " << mut::radiansToDegrees(aa.angle())
                            << std::endl;
                  std::cout << "    Quaternion: [" << QQ.w() << ", (" << QQ.vec().transpose() << ")" << std::endl;
               }

               const cv::Vec3f& euler = mut::rotation2Euler(R);
               std::cout << std::fixed << std::setprecision(4) << "Rotation Roll,Pitch,Yaw: ["
                         << mut::radiansToDegrees(euler[0]) << ","
                         << mut::radiansToDegrees(euler[1]) << "," << mut::radiansToDegrees(euler[2]) << " ]\u00B0 ("
                         << euler[0] << "," << euler[1] << "," << euler[2] << ") radians ("
                         << mut::radiansToDegrees(euler[0]) << "\\textdegree,"
                         << mut::radiansToDegrees(euler[1]) << "\\textdegree," << mut::radiansToDegrees(euler[2])
                         << "\\textdegree)" << std::endl;
               std::cout << std::fixed << std::setprecision(4) << "    Translation: " << TT.at<double>(0, 0) << ","
                         << TT.at<double>(0, 1) << "," << TT.at<double>(0, 2) << std::endl;
               cv::Mat empty_img;
               if (query_img == nullptr)
               {
                  query_img = &empty_img;
                  show_reprojection = false;
               }
               Eigen::Quaterniond Q(aa);
               Eigen::Vector3d t;
               cv::cv2eigen(translations3x1, t);
               std::string name;
               char* pname = nullptr;
               if ((save_reprojection) && (query_img != nullptr))
               {
                  name = "reproject-" + pp.first + ".jpg";
                  pname = const_cast<char*>(name.c_str());
               }
               double max_error, mean_error, stddev_error;
               show_projection(*query_img, world_pts, image_pts, intrinsics, Q, t[0], t[1], t[2],
                               max_error, mean_error, stddev_error, show_reprojection, pname);
               std::cout << std::setprecision(4) << "    Max Error " << max_error << ", mean error " << mean_error
                         << " Std Deviation " << stddev_error << std::endl;
            }
            else
               std::cout << "    No PnP solution" << std::endl;
            std::cout << "---------------------------------------------" << std::endl;
         }
      }
   }
}