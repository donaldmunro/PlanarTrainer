#ifndef _CV_H_4cee9493_5761_45d4_a036_28ca7762dc16
#define _CV_H_4cee9493_5761_45d4_a036_28ca7762dc16

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d.hpp>

namespace ocv
{
   typedef union
   {
      uint32_t class_id;
      struct
      {
         unsigned no            : 16; // Keypoint number
         unsigned imgIdx        : 8; // Training image id
         bool is_selected       : 1; // 1 selected
         bool is_hull           : 1; // 1 if part of convex hull else 0
         bool is_duplicate      : 1;
      };
   } KeyPointExtra_t;

   const int MIN_HOMOGRAPHY_RESULT = 4;

   bool detect(cv::Ptr<cv::Feature2D> detector, cv::Mat &image, std::vector<cv::KeyPoint> &initial_keypoints,
               cv::Mat &descriptors, int top_n = -1);

   bool match(cv::Ptr<cv::DescriptorMatcher>& matcher,
              std::vector<cv::KeyPoint> &train_keypoints, std::vector<cv::KeyPoint> &query_keypoints,
              cv::Mat &train_descriptors, cv::Mat &query_descriptors,
              std::vector<cv::KeyPoint> &matched_train_keypoints, std::vector<cv::KeyPoint> &matched_query_keypoints,
              cv::Mat& matched_train_descriptors, cv::Mat& matched_query_descriptors,
              std::vector<cv::DMatch> &matches, std::vector<std::pair<cv::Point3d, cv::Point3d>>* matched_points = nullptr,
              std::stringstream* errs = nullptr);

   bool homography(const std::vector<cv::KeyPoint>& train_keypoints, const std::vector<cv::KeyPoint>& query_keypoints,
                   const std::vector<cv::DMatch>& matches, cv::Mat& homography, std::vector<cv::DMatch>& homography_matches,
                   std::vector<std::pair<cv::Point3d, cv::Point3d>>* matched_points = nullptr,
                   int method =cv::RHO, float reprojection_error =4);

   bool convex_hull(std::vector<cv::KeyPoint>& keypoints,// const cv::Mat& descriptors,
                    std::vector <cv::Point2f> &hull_points, cv::Point2f &centroid,
                    cv::RotatedRect& rbb, cv::Rect2f& bb,
                    std::vector<cv::KeyPoint>* hull_keypoints =nullptr, //cv::Mat* hull_descriptors = nullptr,
                    std::stringstream *errs = nullptr);

   cv::Mat& draw_matches(const cv::Mat& train_image, const std::vector<cv::KeyPoint>& train_keypoints,
                         const cv::Mat& query_image, const std::vector<cv::KeyPoint>& query_keypoints,
                         const std::vector<cv::DMatch>& matches, cv::Mat& match_img, std::string debug_savename = "",
                         cv::Scalar pointColor =cv::Scalar(0, 0, 255), cv::Scalar lineColor =cv::Scalar(0, 255, 0));

   void convert2x3(const std::vector<cv::KeyPoint>& keypoints, std::vector<cv::Point3d>& points, bool isPoint =true);

   cv::Vec2d project3d(const cv::Mat& K, const Eigen::Quaterniond& Q, const double tx, const double ty, const double tz,
                       const double x, const double y, const double z, Eigen::Vector3d& p);

   void reprojection3d(const Eigen::Matrix3d& K, const Eigen::Matrix3d& R, const Eigen::Vector3d& T,
                       const std::vector<cv::Point3d>& world_pts, const std::vector<cv::Point2d>& image_pts,
                       double& max_error, double &mean_error,
                       const cv::Mat* img = nullptr, cv::Mat* project_image = nullptr,
                       bool draw_query = true,
                       cv::Scalar query_color = cv::Scalar(255, 255, 0),
                       cv::Scalar projection_color = cv::Scalar(0, 255, 0));

   void reprojection2d(const Eigen::Matrix3d& K, const Eigen::Matrix3d& R, const Eigen::Vector3d& T,
                            const std::vector<cv::Point3d>& train_pts, const std::vector<cv::Point3d>& query_pts,
                            const double depth,
                            double& max_error, double& mean_error,
                            const cv::Mat* img = nullptr, cv::Mat* project_image = nullptr,
                            bool draw_query = true,
                            cv::Scalar query_color = cv::Scalar(255, 255, 0),
                            cv::Scalar projection_color = cv::Scalar(0, 255, 0));

   void plot_circles(const cv::Mat& img, double x, double y,
                     const std::vector<cv::Scalar>& colors  ={ cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255),
                                                               cv::Scalar(255, 0, 0),cv::Scalar(0, 255, 255),
                                                               cv::Scalar(0, 0, 255)});

   void plot_rectangles(const cv::Mat& img, double x, double y,
                        const std::vector<cv::Scalar>& colors  ={ cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255),
                                                                  cv::Scalar(255, 0, 0),cv::Scalar(0, 255, 255),
                                                                  cv::Scalar(0, 0, 255)});

   void cvLabel(const cv::Mat& img, const std::string label, const cv::Point &pt, const cv::Scalar background =cv::Scalar(0, 0, 0),
                const cv::Scalar foreground =cv::Scalar(255, 255, 255), const int font = cv::FONT_HERSHEY_DUPLEX);
}
#endif //_CV_H_4cee9493_5761_45d4_a036_28ca7762dc16
