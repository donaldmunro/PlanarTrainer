#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <random>
#include <cmath>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


#include "display.h"
#include "pose/poseocv.h"

cv::Vec2d project3d(const cv::Mat& K, const Eigen::Quaterniond& Q, const double tx, const double ty, const double tz,
                    const double x, const double y, const double z, Eigen::Vector3d& p)
//-----------------------------------------------------------------------------------
{
   Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> EK((double *) K.data);
//   Eigen::Quaterniond QR = Q*Eigen::Quaterniond(0, x, y, z)*Q.inverse();
//   p = EK*(QR.vec() + Eigen::Vector3d(tx, ty, tz));
//   p /= p[2];
   p = EK*(Q*Eigen::Vector3d(x, y, z) + Eigen::Vector3d(tx, ty, tz));
   p /= p[2];
   return cv::Vec2d(p[0], p[1]);
}

void show_projection(const cv::Mat& img, const std::vector<cv::Point3d>& world_pts,
                     const std::vector<cv::Point3d>& image_pts, const cv::Mat& K,
                     const Eigen::Quaterniond& Q, const double tx, const double ty, const double tz,
                     double& max_error, double &mean_error, double &stddev_error, bool show, const char* save_img_name)
//--------------------------------------------------------------------------------------------------------------------
{
   cv::Mat project_img;
   if (! img.empty())
      img.copyTo(project_img);
   std::vector<cv::Scalar> black(7), cyan(7);
   std::fill(black.begin(), black.begin()+7, cv::Scalar(0, 0, 0));
   std::fill(cyan.begin(), cyan.begin()+7, cv::Scalar(255,255,0));
   max_error = std::numeric_limits<double >::min(), mean_error = 0;
   size_t n = std::min(world_pts.size(), image_pts.size());
   std::vector<double> errors;
   for (size_t i=0; i<n; i++)
   {
      cv::Point3d qpt = image_pts[i];
      if (! img.empty())
         plot_rectangles(project_img, qpt.x, qpt.y, cyan);
      cv::Point3d wpt = world_pts[i];
      Eigen::Vector3d ept;
      const cv::Vec2d pt = project3d(K, Q, tx, ty, tz, wpt.x, wpt.y, wpt.z, ept);
      if (! img.empty())
         plot_circles(project_img, pt[0], pt[1], black);
      double error = cv::norm(pt - cv::Vec2d(qpt.x, qpt.y));
      if (error > max_error)
         max_error = error;
      mean_error += error;
      errors.push_back(error);
   }
   mean_error /= n;
   stddev_error = 0;
   for (double error : errors)
      stddev_error += std::pow(mean_error - error, 2);
   stddev_error /= n;
   if (! img.empty())
   {
      if (save_img_name != nullptr)
         cv::imwrite(save_img_name, project_img);
      if (show)
         cv::imshow("ReProjection", project_img);
   }
}

void project2d(const Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>& EK,
               const Eigen::Matrix3d& R, const Eigen::Vector3d& T,
               const std::vector<cv::Point3d>& train_pts, const std::vector<cv::Point3d>& query_pts,
               const double depth, cv::Mat* project_image, double& max_error, double& mean_error, bool draw_query)
//----------------------------------------------------------------------------------------------------------------
{
   max_error = std::numeric_limits<double >::min(); mean_error = 0;
   std::vector<cv::Scalar> black(7), cyan(7);
   std::fill(black.begin(), black.begin()+7, cv::Scalar(0, 0, 0));
   std::fill(cyan.begin(), cyan.begin()+7, cv::Scalar(255,255,0));
   for (size_t j=0; j<train_pts.size(); j++)
   {
      cv::Point3d qpt = query_pts[j];
      if ( (project_image != nullptr) && (! project_image->empty()) && (draw_query) )
         plot_rectangles(*project_image, qpt.x, qpt.y, cyan);
      cv::Point3d tpt = train_pts[j];
      Eigen::Vector3d ray = EK.inverse()*Eigen::Vector3d(tpt.x, tpt.y, 1);
      ray /= ray[2];
      if (! std::isnan(depth))
         ray *= depth;
      Eigen::Vector3d ipt = R*ray + T;
//      std::cout << (R*ray).transpose() << " + " << T.transpose() << " = " << ipt.transpose() << std::endl;
      ipt = EK*ipt;
      ipt /= ipt[2];
      if ( (project_image != nullptr) && (! project_image->empty()) )
         plot_circles(*project_image, ipt[0], ipt[1], black);

      double error = (ipt - Eigen::Vector3d(qpt.x, qpt.y, qpt.z)).norm();
      if (error > max_error)
         max_error = error;
      mean_error += error;
//     std::cout << j << ": " << qpt << " " << ipt.transpose() << " = " << error << std::endl;
   }
   mean_error /= train_pts.size();
}

void noise(const double deviation, std::vector<cv::Point3d>& query_pts, bool isRoundInt, bool isX, bool isY, bool isZ,
           std::vector<cv::Point3d>* destination)
//-------------------------------------------------------------------------------------------------------
{
   std::random_device rd{};
   std::mt19937 gen{rd()};
   std::normal_distribution<double> N{0,deviation};
   if (destination != nullptr)
      destination->clear();
   for (cv::Point3d& pt : query_pts)
   {
      double& x = pt.x;
      double& y = pt.y;
      double& z = pt.z;
      double xnoise =0, ynoise =0, znoise =0;
      if (isX)
         xnoise = N(gen);
      if (isY)
         ynoise = N(gen);
      if (isZ)
         znoise = N(gen);
      if (destination != nullptr)
      {
         if (isRoundInt)
            destination->emplace_back(std::round(x + xnoise), std::round(y + ynoise), std::round(z + znoise));
         else
            destination->emplace_back(x + xnoise, y + ynoise, z + znoise);
      }
      else
      {
         x += xnoise;
         y += ynoise;
         z += znoise;
         if (isRoundInt)
         {
            x = std::round(x);
            y = std::round(y);
            z = std::round(z);
         }
      }
   }

}
