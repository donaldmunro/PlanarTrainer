#ifndef GRAVITYPOSE_POSEOCV_H
#define GRAVITYPOSE_POSEOCV_H

#include <type_traits>
#include <chrono>
#include <array>

#include <Eigen/Core>

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d.hpp>

#include "Ransac.hh"
#include "two_point_pose_partial_rotation.h"

using TimeType = std::chrono::_V2::system_clock::time_point;

namespace poseocv
{
   struct PnPRANSACParameters
   {
      int  	   iterationsCount = 100;
      float  	reprojectionError = 8.0;
      double  	confidence = 0.99;
   };

   void homography_pose(const cv::Mat& intrinsics, const std::vector<cv::Point3d>& train_img_pts,
                        const std::vector<cv::Point3d>& query_img_pts, std::vector<cv::Mat>& rotations,
                        std::vector<cv::Mat>& translations, std::vector<cv::Mat>& normals, bool isRANSAC = false);

   bool pose_PnP(const std::vector<cv::Point3d>& world_pts, const std::vector<cv::Point3d>& image_pts,
                 const cv::Mat& intrinsics, cv::Mat& rotation_vec, cv::Mat& translations3x1, cv::Mat* R,
                 long& total_time, bool is_refine = false, int flags = cv::SOLVEPNP_ITERATIVE,
                 PnPRANSACParameters* RANSACParams = nullptr);

   void display_PnP(const std::vector<cv::Point3d>& world_pts,
                    const std::vector<cv::Point3d>& image_pts,
                    const cv::Mat& intrinsics, bool isMinimal = false, const cv::Mat* query_img = nullptr,
                    bool show_reprojection = false, bool save_reprojection = true, bool is_time = true);
}
#endif //GRAVITYPOSE_POSEOCV_H
