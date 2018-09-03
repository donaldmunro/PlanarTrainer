#include <iostream>
#include <limits>

#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Eigen>
#include <unsupported/Eigen/NonLinearOptimization>

#include <opencv2/calib3d/calib3d.hpp>
#include <sys/stat.h>

#include "Optimization.h"
#include "math.hh"

bool translation_gauss_newton(const std::vector<cv::Point3d>& world_pts, const std::vector<cv::Point2d>& image_pts,
                              const Eigen::Matrix3d &K, const Eigen::Quaterniond& Q, Eigen::Vector3d &T,
                              double& residual, int& iter)
//----------------------------------------------------------------------------------------------------------------------
{
   double tx = T[0], ty = T[1], tz = T[2];
   const Eigen::Matrix3d KI = K.inverse();
   size_t m = std::min(world_pts.size(), image_pts.size());
   std::vector<Eigen::Quaterniond> world_quaternions;
   std::vector<Eigen::Vector3d> image_rays;
   const Eigen::Quaterniond QI = Q.inverse();
   for (size_t row = 0; row < m; row++)
   {
      cv::Point2d &pt = const_cast<cv::Point2d &>(image_pts[row]);
      Eigen::Vector3d pt2d = KI * Eigen::Vector3d(pt.x, pt.y, 1);
      pt2d /= pt2d[2];
      image_rays.push_back(pt2d);
      const cv::Point3d pt3d = world_pts[row];
      const Eigen::Quaterniond QR = Q * Eigen::Quaterniond(0, pt3d.x, pt3d.y, pt3d.z) * QI;
      world_quaternions.push_back(QR);
   }
   iter = 0;
   double prev_min_residual = std::numeric_limits<double>::max();
   double eps = 0.0000001;
   do
   {
      Eigen::MatrixXd J(m, 3);
//      Eigen::MatrixXd r(m, 3);
      Eigen::MatrixXd r(m, 1);
      double min_residual = std::numeric_limits<double>::max();
      for (size_t row = 0; row < m; row++)
      {
         Eigen::Vector3d pt2d = image_rays[row];
         double u = pt2d[0], v = pt2d[1];
         const Eigen::Vector3d Rv = world_quaternions[row].vec();
         //  Eigen::Vector3d Rr = R*Eigen::Vector3d(pt3d.x, pt3d.y, pt3d.z);
         Eigen::Vector3d pt3d = Rv + Eigen::Vector3d(tx, ty, tz);
         pt3d /= pt3d[2];
         Eigen::Vector3d diff = pt3d - pt2d;
         residual = diff.dot(diff);
         if (residual < min_residual)
            min_residual = residual;
//         std::cout << "refine: " << pt3d.transpose() << " " << pt2d.transpose() << " " << diff.transpose() << " " << residual << std::endl;
         Eigen::Vector3d d;
         J.row(row) << 2 * (Rv[0] + tx - u), 2 * (Rv[1] + ty - v), 2 * (Rv[2] + tz - 1);
         r.row(row) << residual;
      }
//      std::cout << "minmax " << min_residual << " " << (min_residual - prev_min_residual) <<  std::endl;
      if (min_residual > prev_min_residual) break;
      prev_min_residual = min_residual;
      auto Jt = J.transpose();
//      std::cout << "JTr: " << std::endl << (Jt * r) << std::endl << "==================== " << std::endl;

      //auto llt = (Jt * J).ldlt();
      auto llt = (Jt * J).llt();
      double dx, dy, dz;
      if (llt.info() == Eigen::Success)
      {
         auto delta =  llt.solve(Jt * r * (-1.0));
         dx = delta(0, 0), dy = delta(1, 0), dz = delta(2, 0);
//         std::cout << delta << std::endl;
      }
      else
      {
         auto JtJI = (Jt*J).inverse();
         auto dd = -(JtJI*Jt);
         auto delta = dd*r;
         dx = delta(0, 0), dy = delta(1, 0), dz = delta(2, 0);
      }
      if ( (mut::near_zero(dx, eps)) && (mut::near_zero(dy, eps)) && (mut::near_zero(dz, eps)) ) break;
      tx += dx;
      ty += dy;
      tz += dz;
   } while (iter++ < 200);
   if ( (iter > 1) && (iter < 100) )
   {
      T[0] = tx; T[1] = ty; T[2] = tz;
      return true;
   }
   return false;
}

bool translation_levenberg_marquardt3d(const std::vector<cv::Point3d>& world_pts, const std::vector<cv::Point2d>& image_pts,
                                     const Eigen::Matrix3d &K, const Eigen::Quaterniond& Q, Eigen::Vector3d &T,
                                     int& iterations)
//-----------------------------------------------------------------------------------------------------------------------
{
   TranslationLevenbergMarquardt3D functor(world_pts, image_pts, K, Q);
   Eigen::LevenbergMarquardt<TranslationLevenbergMarquardt3D, double> lm(functor);
   Eigen::VectorXd x(3);
   x << T[0], T[1], T[2];
   iterations = 0;
   Eigen::LevenbergMarquardtSpace::Status status = lm.minimizeInit(x);
   if (status == Eigen::LevenbergMarquardtSpace::ImproperInputParameters)
      return false;
   do
   {
      status = lm.minimizeOneStep(x);
      if (status == Eigen::LevenbergMarquardtSpace::Running)
         iterations++;
   } while (status == Eigen::LevenbergMarquardtSpace::Running);
   if (iterations > 1)
   {
      T = x;
      return true;
   }
   return false;
}

bool translation_levenberg_marquardt2d_depth(const std::vector<cv::Point3d>& world_pts,
                                             const std::vector<cv::Point3d>& image_pts, const Eigen::Matrix3d &K,
                                             const Eigen::Quaterniond& Q, Eigen::Vector3d &T, const double depth,
                                             int& iterations)
//--------------------------------------------------------------------------------------------------------------
{
   TranslationLevenbergMarquardt2DDepth functor(world_pts, image_pts, K, Q, depth);
   Eigen::LevenbergMarquardt<TranslationLevenbergMarquardt2DDepth, double> lm(functor);
   Eigen::VectorXd x(3);
   x << T[0], T[1], T[2];
   iterations = 0;
   Eigen::LevenbergMarquardtSpace::Status status = lm.minimizeInit(x);
   if (status == Eigen::LevenbergMarquardtSpace::ImproperInputParameters)
      return false;
   do
   {
      status = lm.minimizeOneStep(x);
      if (status == Eigen::LevenbergMarquardtSpace::Running)
         iterations++;
   } while (status == Eigen::LevenbergMarquardtSpace::Running);
   if (iterations > 1)
   {
      T = x;
      return true;
   }
   return false;
}