#ifndef GRAVITYPOSE_POSE2D_H
#define GRAVITYPOSE_POSE2D_H

#include <opencv2/core/core.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace pose2d
{
   const double PI = 3.14159265358979323846;

   void pose(const std::vector<std::pair<cv::Point3d, cv::Point3d>>& pts,
             const Eigen::Vector3d& train_g, const Eigen::Vector3d query_g,
             const cv::Mat& intrinsics, Eigen::Quaterniond& Q, Eigen::Vector3d& translation);

   void pose(const std::vector<std::pair<cv::Point3d, cv::Point3d>>& pts,
             const Eigen::Vector3d& train_g, const Eigen::Vector3d query_g,
             const Eigen::Matrix3d& K, Eigen::Quaterniond& Q, Eigen::Vector3d& translation);

   void pose(const std::vector<std::pair<cv::Point3d, cv::Point3d>>& pts,
             const Eigen::Vector3d& train_g, const Eigen::Vector3d query_g, const double depth,
             const cv::Mat& intrinsics, Eigen::Quaterniond& Q, Eigen::Vector3d& translation);

   void pose(const std::vector<std::pair<cv::Point3d, cv::Point3d>>& pts,
             const Eigen::Vector3d& train_g, const Eigen::Vector3d query_g, const double depth,
             const Eigen::Matrix3d& K, Eigen::Quaterniond& Q, Eigen::Vector3d& translation);

   double pose_ransac(const std::vector<std::pair<cv::Point3d, cv::Point3d>>& pts,
                      const Eigen::Vector3d& train_g, const Eigen::Vector3d query_g,
                      const cv::Mat& intrinsics, Eigen::Quaterniond& Q, Eigen::Vector3d& translation,
                      void *RANSAC_params, int samples);

   double pose_ransac(const std::vector<std::pair<cv::Point3d, cv::Point3d>>& pts,
                      const Eigen::Vector3d& train_g, const Eigen::Vector3d query_g,
                      const Eigen::Matrix3d& K, Eigen::Quaterniond& Q, Eigen::Vector3d& translation,
                      void *RANSAC_params, int samples);

   double pose_ransac(const std::vector<std::pair<cv::Point3d, cv::Point3d>>& pts,
                      const Eigen::Vector3d& train_g, const Eigen::Vector3d query_g,
                      const double depth, const cv::Mat& intrinsics,
                      Eigen::Quaterniond& Q, Eigen::Vector3d& translation,
                      void *RANSAC_params, int samples);

   double pose_ransac(const std::vector<std::pair<cv::Point3d, cv::Point3d>>& pts,
                      const Eigen::Vector3d& train_g, const Eigen::Vector3d query_g,
                      const double depth, const Eigen::Matrix3d& K,
                      Eigen::Quaterniond& Q, Eigen::Vector3d& translation,
                      void *RANSAC_params, int samples);

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

#endif //GRAVITYPOSE_POSE2D_H
