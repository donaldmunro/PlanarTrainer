#ifndef GRAVITYPOSE_POSE3D_H
#define GRAVITYPOSE_POSE3D_H

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/core/core.hpp>

#include "PoseRANSAC.hh"

namespace pose3d
{
   const double PI = 3.14159265358979323846;

   bool pose(const std::vector<std::pair<cv::Point3d, cv::Point2d>>& pts,
             const Eigen::Vector3d& train_g, const Eigen::Vector3d query_g,
             const Eigen::Matrix3d& KI, Eigen::Quaterniond& Q, Eigen::Vector3d& translation);

//Refined with Levenberg/Marquardt
   void refine(const std::vector<cv::Point3d>& world_pts, const std::vector<cv::Point2d>& query_image_pts,
               const Eigen::Matrix3d& K, Eigen::Quaterniond& Q, Eigen::Vector3d& translation, int& iterations);

   double pose_ransac(const std::vector<std::pair<cv::Point3d, cv::Point2d>>& pts,
                      const Eigen::Vector3d& train_g, const Eigen::Vector3d query_g,
                      const Eigen::Matrix3d& KI, Eigen::Quaterniond& Q, Eigen::Vector3d& translation,
                      void *RANSAC_params, int samples);

   void pose_translation(const std::vector<cv::Point3d>& world_pts, const std::vector<cv::Point2d>& query_image_pts,
                         const Eigen::Matrix3d& KI, const Eigen::Quaterniond& Q, Eigen::Vector3d& translation);

   void pose_translation(const std::vector<cv::Point3d>& world_pts, const std::vector<cv::Point2d>& query_image_pts,
                         const Eigen::Matrix3d& KI, const Eigen::Matrix3d& R, Eigen::Vector3d& translation);
};
#endif //GRAVITYPOSE_POSE3D_H
