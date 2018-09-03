#ifndef GRAVITYPOSE_OPTIMIZATION_H
#define GRAVITYPOSE_OPTIMIZATION_H

#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/core/core.hpp>

bool translation_gauss_newton(const std::vector<cv::Point3d>& world_pts, const std::vector<cv::Point2d>& image_pts,
                              const Eigen::Matrix3d &K, const Eigen::Quaterniond& Q, Eigen::Vector3d &T,
                              double& residual, int& iter);

bool translation_levenberg_marquardt3d(const std::vector<cv::Point3d>& world_pts, const std::vector<cv::Point2d>& image_pts,
                                       const Eigen::Matrix3d &K, const Eigen::Quaterniond& Q, Eigen::Vector3d &T,
                                       int& iterations);
struct TranslationLevenbergMarquardt3D
//====================================
{
   TranslationLevenbergMarquardt3D(const std::vector<cv::Point3d>& world_pts_, const std::vector<cv::Point2d>& image_pts_,
                                   const Eigen::Matrix3d &K, const Eigen::Quaterniond& Q_)
   :  world_pts(world_pts_), Q(Q_), QI(Q_.inverse()), KI(K.inverse()),
      m(static_cast<int>(std::min(world_pts_.size(), image_pts_.size())))
      //----------------------------------------------------------------------------------------------------------------
      {
         for (size_t row = 0; row < m; row++)
         {
            cv::Point2d pt = image_pts_[row];
            Eigen::Vector3d pt2d = KI * Eigen::Vector3d(pt.x, pt.y, 1);
            pt2d /= pt2d[2];
            image_rays.push_back(pt2d);
            const cv::Point3d pt3d = world_pts[row];
            const Eigen::Quaterniond QR = Q * Eigen::Quaterniond(0, pt3d.x, pt3d.y, pt3d.z) * QI;
            //  Eigen::Vector3d Rr = R*Eigen::Vector3d(pt3d.x, pt3d.y, pt3d.z);
            rotated_world_pts.push_back(QR.vec());
         }
      }

   // Compute 'm' errors, one for each data point, for the given parameter values in 'x'
   int operator()(const Eigen::VectorXd& T, Eigen::VectorXd& residuals) const
   //-------------------------------------------------------------------
   {
      for (size_t row = 0; row < m; row++)
      {
         Eigen::Vector3d pt2d = image_rays[row];
         const Eigen::Vector3d rpt3d = rotated_world_pts[row];
         Eigen::Vector3d pt3d = rpt3d + Eigen::Vector3d(T[0], T[1], T[2]);
         pt3d /= pt3d[2];
         Eigen::Vector3d diff = pt3d - pt2d;
         residuals(row) = diff.dot(diff);
      }
//      std::cout << "Residuals: " <<  residuals.transpose() << std::endl;
      return 0;
   }

   // Compute the jacobian of the errors
   int df(const Eigen::VectorXd& T, Eigen::MatrixXd& J) const
   //-----------------------------------------------------------
   {
//      std::cout << "T = " << T.transpose() << std::endl;
      for (size_t row = 0; row < m; row++)
      {
         Eigen::Vector3d pt2d = image_rays[row];
         const Eigen::Vector3d pt3d = rotated_world_pts[row];
         J.row(row) << 2 * (pt3d[0] + T[0] - pt2d[0]), 2 * (pt3d[1] + T[1] - pt2d[1]), 2 * (pt3d[2] + T[2] - 1);
      }
//      std::cout << "J = " << std::endl << J << std::endl;
      return 0;
   }

   size_t values() const { return m; }

   size_t inputs() const { return 3; }

   const std::vector<cv::Point3d>& world_pts;
   const Eigen::Quaterniond& Q;
   const Eigen::Quaterniond QI;
   const Eigen::Matrix3d KI;
   size_t m;
   std::vector<Eigen::Vector3d> image_rays;
   std::vector<Eigen::Vector3d> rotated_world_pts;
};

bool translation_levenberg_marquardt2d_depth(const std::vector<cv::Point3d>& world_pts,
                                             const std::vector<cv::Point3d>& image_pts, const Eigen::Matrix3d &K,
                                             const Eigen::Quaterniond& Q, Eigen::Vector3d &T, const double depth,
                                             int& iterations);
struct TranslationLevenbergMarquardt2DDepth
//=========================================
{
   TranslationLevenbergMarquardt2DDepth(const std::vector<cv::Point3d>& train_points,
                                        const std::vector<cv::Point3d>& query_points,
                                        const Eigen::Matrix3d &K, const Eigen::Quaterniond& Q_, const double d)
   :  R(Q_.toRotationMatrix()), KI(K.inverse()), m(static_cast<int>(std::min(train_points.size(), query_points.size()))),
      depth(d)
   //----------------------------------------------------------------------------------------------------------
   {
      for (size_t row = 0; row < m; row++)
      {
         cv::Point3d pt = train_points[row];
         Eigen::Vector3d pt2d = KI * Eigen::Vector3d(pt.x, pt.y, 1);
         pt2d /= pt2d[2];
         pt2d *= depth;
         Eigen::Vector3d ipt = R*pt2d;
         rotated_train_pts.push_back(ipt);

         pt = query_points[row];
         Eigen::Vector3d qpt = KI*Eigen::Vector3d(pt.x, pt.y, 1);
         query_pts.emplace_back(qpt);
      }
   }
   
   // Compute 'm' errors, one for each data point, for the given parameter values in 'x'
   int operator()(const Eigen::VectorXd& T, Eigen::VectorXd& residuals) const
   //-------------------------------------------------------------------
   {
      for (size_t row = 0; row < m; row++)
      {
         Eigen::Vector3d train_pt = rotated_train_pts[row];
         train_pt += T;
         train_pt /= train_pt[2];
         Eigen::Vector3d query_pt = query_pts[row];
         Eigen::Vector3d diff = query_pt - train_pt;
         residuals(row) = diff.dot(diff);
      }
      //      std::cout << "Residuals: " <<  residuals.transpose() << std::endl;
      return 0;
   }
   
   // Compute the jacobian of the errors
   int df(const Eigen::VectorXd& T, Eigen::MatrixXd& J) const
   //-----------------------------------------------------------
   {
      //      std::cout << "T = " << T.transpose() << std::endl;
      for (size_t row = 0; row < m; row++)
      {
         Eigen::Vector3d pt2d = query_pts[row];
         const Eigen::Vector3d pt3d = rotated_train_pts[row];
         J.row(row) << 2 * (pt3d[0] + T[0] - pt2d[0]), 2 * (pt3d[1] + T[1] - pt2d[1]), 2 * (pt3d[2] + T[2] - 1);
      }
      //      std::cout << "J = " << std::endl << J << std::endl;
      return 0;
   }

   size_t values() const { return m; }
   
   int inputs() const { return 3; }
   
   const Eigen::Matrix3d R;
   const Eigen::Matrix3d KI;
   size_t m;
   double depth;
   std::vector<Eigen::Vector3d> rotated_train_pts;
   std::vector<Eigen::Vector3d> query_pts;
};

#endif //GRAVITYPOSE_OPTIMIZATION_H
