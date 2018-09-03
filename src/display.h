#ifndef GRAVITYPOSE_UT_H
#define GRAVITYPOSE_UT_H

#include <string>
#include <vector>
#include <random>

#include <Eigen/Core>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "pose/math.hh"

cv::Vec2d project3d(const cv::Mat& K, const Eigen::Quaterniond& Q, const double tx, const double ty, const double tz,
                    const double x, const double y, const double z, Eigen::Vector3d& p);

void show_projection(const cv::Mat& img, const std::vector<cv::Point3d>& world_pts,
                     const std::vector<cv::Point3d>& image_pts, const cv::Mat& K,
                     const Eigen::Quaterniond& Q, const double tx, const double ty, const double tz,
                     double& max_error, double &mean_error, double &stddev_error, bool show =true,
                     const char* save_img_name = nullptr);

static int show_images() { return 0; }

void project2d(const Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>& EK,
               const Eigen::Matrix3d& R, const Eigen::Vector3d& T,
               const std::vector<cv::Point3d>& train_pts, const std::vector<cv::Point3d>& query_pts,
               const double depth, cv::Mat* project_image, double& max_error, double& mean_error, bool draw_query =true);

void noise(const double deviation, std::vector<cv::Point3d>& query_pts, bool isRoundInt, bool isX =true, bool isY =true,
           bool isZ =false, std::vector<cv::Point3d>* destination = nullptr);

template <typename ...Args>
int show_images(const char* title, const cv::Mat& img, int offset, Args... args)
//-------------------------------------------------------------------------------
{
   int n = show_images(args...);
   if (! img.empty())
   {
      cv::namedWindow(title);
      cv::imshow(title, img);
      cv::moveWindow(title, offset, 0);
      return n + 1;
   }
   else
      return n;
}

inline void plot_circles(const cv::Mat& img, double x, double y,
                         const std::vector<cv::Scalar>& colors  ={ cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255),
                                                                   cv::Scalar(255, 0, 0),cv::Scalar(0, 255, 255),
                                                                   cv::Scalar(0, 0, 255)})
//----------------------------------------------------------------------------------------------------------------
{
   int i = 2;
   for (cv::Scalar color : colors)
   {
      cv::circle(img, cv::Point2i(cvRound(x), cvRound(y)), i, color, 1);
      i++;
   }
}

inline void plot_rectangles(const cv::Mat& img, double x, double y,
                            const std::vector<cv::Scalar>& colors  ={ cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255),
                                                                      cv::Scalar(255, 0, 0),cv::Scalar(0, 255, 255),
                                                                      cv::Scalar(0, 0, 255)})
//--------------------------------------------------------------------------------------------------------
{
   int i = 1;
   for (cv::Scalar color : colors)
   {
      cv::rectangle(img, cv::Point2i(cvRound(x) - i, cvRound(y) - i), cv::Point2i(cvRound(x) + i, cvRound(y) + i),
                    color, 1);
      i++;
   }
}
#endif
