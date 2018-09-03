#pragma once
#ifndef _MATHUTIL_H
#define _MATHUTIL_H

#include <assert.h>

#include <random>
#include <limits>
#include <memory>
#include <iostream>
#include <iomanip>

#include <Eigen/src/Core/util/DisableStupidWarnings.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d.hpp>

namespace mut
{
   const double PI = 3.14159265358979323846;
   const float PI_F = 3.14159265358979f;
   const double NaN = std::numeric_limits<double>::quiet_NaN();
   const double NaN_F = std::numeric_limits<float>::quiet_NaN();

   static Eigen::Quaterniond rotate_between(const Eigen::Vector3d &from, const Eigen::Vector3d &to,
                                            const Eigen::Vector3d &fallbackAxis = Eigen::Vector3d(0, 0, 0))
   //-----------------------------------------------------------------------------------------------
   {
      Eigen::Quaterniond q;
      Eigen::Vector3d v0 = from;
      Eigen::Vector3d v1 = to;
      v0.normalize();
      v1.normalize();

      double d = v0.dot(v1);
      if (d >= 1.0f)
         return Eigen::Quaterniond(1, 0, 0, 0);

      if (d < (1e-6f - 1.0f))
      {
         if (fallbackAxis != Eigen::Vector3d(0, 0, 0))
            q = Eigen::AngleAxis<double>(PI, fallbackAxis);
         else
         {
            // Generate an axis
            Eigen::Vector3d axis = Eigen::Vector3d(1, 0, 0).cross(from);
            if (axis.norm() < 0.000000001) // pick another if colinear
               axis = Eigen::Vector3d(0, 1, 0).cross(from);
            axis.normalize();
            q = Eigen::AngleAxis<double>(PI, axis);
         }
      }
      else
      {
         double s = sqrt((1 + d) * 2);
         double invs = 1 / s;

         Eigen::Vector3d c = v0.cross(v1);

         q.x() = c.x() * invs;
         q.y() = c.y() * invs;
         q.z() = c.z() * invs;
         q.w() = s * 0.5f;
         q.normalize();
      }
      return q;
   }

   template<typename T> static inline bool near_zero(T v, T epsilon);

   template<> bool near_zero(long double v, long double epsilon) { return (fabsl(v) <= epsilon); }

   template<> bool near_zero(double v, double epsilon) { return (fabs(v) <= epsilon); }

   template<> bool near_zero(float v, float epsilon) { return (fabsf(v) <= epsilon); }

   inline static double radiansToDegrees(double radians) { return (radians * 180.0 / PI); }

   inline static float radiansToDegrees(float radians) { return (radians * 180.0f / PI_F); }

   inline static double degreesToRadians(double degrees) { return (degrees * PI / 180.0); }

   inline static float degreesToRadians(float degrees) { return (degrees * PI_F / 180.0f); }

   template<typename T>
   inline static void normalize(T &v)
//-----------------------------------------------
   {
      auto magnitude = v.dot(v);
      if (magnitude > 1.0e-8)
         v /= sqrt(magnitude);
   }

   template<typename T>
   inline static void unchecked_normalize(T &v)
   { v /= sqrt(v.dot(v)); }

   inline static Eigen::Matrix3d skew33d(Eigen::Vector3d v3)
  //-----------------------------------------------
   {
      double a = v3[0], b = v3[1], c = v3[2];
      Eigen::Matrix3d X;
      X << 0, -c, b, c, 0, -a, -b, a, 0;
      return X;
   }

   inline static Eigen::Vector3d solve_homogeneous(Eigen::Matrix3d A)
//---------------------------------------------------------------------------------------------------------------------------
   {
//   std::cout << "Ax = 0:" << std::endl << A << std::endl;
      Eigen::JacobiSVD<Eigen::MatrixXd, Eigen::FullPivHouseholderQRPreconditioner>
            svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
      auto U = svd.matrixU();
      auto V = svd.matrixV();
//   std::cout << "U: " << U << std::endl << "S: " << svd.singularValues() << std::endl << "V: " << V << std::endl;
      return V.col(V.cols() - 1);
   }


   inline static Eigen::Vector3d ray(const Eigen::Matrix3d &KI, double x, double y, double z = 1,
                                                bool is_homogenouous = false, bool is_normalize = false)
//----------------------------------------------------------------------------------
   {
      Eigen::Vector3d v(x, y, z);
      Eigen::Vector3d ray = KI * v;
      if (is_homogenouous)
         ray /= ray[2];
      if (is_normalize)
         ray.normalize();
      return ray;
   }

   inline static Eigen::Vector3d ray(const Eigen::Matrix3d &KI, Eigen::Vector3d v, bool is_homogenouous = false,
                                                bool is_normalize = false)
//--------------------------------------------------------------------------------------------------
   {
      Eigen::Vector3d ray = KI * v;
      if (is_homogenouous)
         ray /= ray[2];
      if (is_normalize)
         ray.normalize();
      //Eigen::Vector3d p  = KI.inverse()*Eigen::Vector3d(1128,592);
      return ray;
   }

   inline static Eigen::Vector3d ray(const Eigen::Matrix3d &KI, cv::Point3d pt, bool is_homogenouous = false,
                                                bool is_normalize = false)
   //-----------------------------------------------------------------------------------------------------------------------
   {
      Eigen::Vector3d v(pt.x, pt.y, pt.z);
      Eigen::Vector3d ray = KI * v;
      if (is_homogenouous)
         ray /= ray[2];
      if (is_normalize)
         ray.normalize();
      return ray;
   }

   inline static void yawpitchroll(const Eigen::Vector3d train, const Eigen::Vector3d query,
                                   double& roll_degrees, double& pitch_degrees, double &yaw_degrees)
   //----------------------------------------------------------------------------------------
   {
      Eigen::Quaterniond Q = rotate_between(query, train);
      Eigen::Matrix3d R = Q.toRotationMatrix();
      Eigen::Vector3d euler = R.eulerAngles(0, 1, 2);
      roll_degrees = radiansToDegrees(euler[0]);
      pitch_degrees = radiansToDegrees(euler[1]);
      yaw_degrees = radiansToDegrees(euler[2]);

      std::cout << std::fixed << std::setprecision(8) << train.transpose() << " -> "  << query.transpose()
                << ": Roll " << mut::radiansToDegrees(euler[0]) << " Pitch " << mut::radiansToDegrees(euler[1])
                << " Yaw " << mut::radiansToDegrees(euler[2]) << std::endl << R << std::endl;

      Q.setFromTwoVectors(query, train);
      R = Q.toRotationMatrix();
      euler = R.eulerAngles(0, 1, 2);
      std::cout << std::fixed << std::setprecision(8) << train.transpose() << " -> "  << query.transpose()
                << ": Roll " << mut::radiansToDegrees(euler[0]) << " Pitch " << mut::radiansToDegrees(euler[1])
                << " Yaw " << mut::radiansToDegrees(euler[2]) << std::endl << R << std::endl;
   }

   static Eigen::Vector3d quaternion_to_euler_angle(Eigen::Quaterniond Q)
   //--------------------------------------------------------------------
   {
      double w = Q.w();
      double x = Q.x();
      double y = Q.y();
      double z = Q.z();

      double ysqr = y * y;

      double t0 = +2.0 * (w * x + y * z);
      double t1 = +1.0 - 2.0 * (x * x + ysqr);
      double X = atan2(t0, t1);

      double t2 = +2.0 * (w * y - z * x);
      t2 = (t2 > +1.0) ? +1.0 : t2;
      t2 = (t2 < -1.0) ? -1.0 : t2;
      double Y = asin(t2);

      double t3 = +2.0 * (w * z + x * y);
      double t4 = +1.0 - 2.0 * (ysqr + z * z);
      double Z = atan2(t3, t4);

      return Eigen::Vector3d(X, Y, Z);
   }

   static cv::Vec3f rotation2Euler(cv::Mat &R)
   //-----------------------------------------
   {
      Eigen::Matrix3d ER;
      cv::cv2eigen(R, ER);
      Eigen::Vector3d euler = ER.eulerAngles(0, 1, 2);
      return cv::Vec3f(euler[0], euler[1], euler[2]);
   }

   static Eigen::Quaterniond euler2Quaternion(const double roll, const double pitch, const double yaw )
   //-------------------------------------------------------------------------------------------
   {
      Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitX());
      Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitY());
      Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitZ());
      Eigen::Quaterniond q = yawAngle * pitchAngle * rollAngle;
      return q;
   }

   static cv::Vec3d ocvMultMatVec(const cv::Mat &P, double x, double y, double z)
   //---------------------------------------------------------------------------
   {
      cv::Mat V(1, 3, CV_64FC1);
      double *pV = V.ptr<double>(0);
      pV[0] = x;
      pV[1] = y;
      pV[2] = z;
      cv::Mat R = P * V.t();
      R = R.t();
      pV = R.ptr<double>(0);
      return cv::Vec3d(pV[0], pV[1], pV[2]);
   }

   static cv::Vec3d ocvMultMatVec(const cv::Mat &P, cv::Vec3d v)
   //------------------------------------------------------------
   {
      cv::Mat V(1, 3, CV_64FC1);
      double *pV = V.ptr<double>(0);
      pV[0] = v[0];
      pV[1] = v[1];
      pV[2] = v[2];
      cv::Mat R = P * V.t();
      R = R.t();
      pV = R.ptr<double>(0);
      return cv::Vec3d(pV[0], pV[1], pV[2]);
   }

   static inline cv::Vec3d toOcvRotationVec(Eigen::Quaterniond Q)
   //-----------------------------------------------------
   {
      Eigen::AngleAxisd aa(Q);
      Eigen::Vector3d v = aa.angle()*aa.axis();
/*
      auto R = Q.toRotationMatrix();
      cv::Mat OR;
      cv::eigen2cv(R, OR);
      cv::Vec3d vv;
      cv::Rodrigues(OR, vv);
      std::cout << v.transpose() << " " << vv.t() << std::endl;
*/
      return cv::Vec3d(v[0], v[1], v[2]);
   }

   static inline Eigen::Quaterniond toQuaternion(cv::InputArray m)
   //-----------------------------------------------------
   {
      cv::Mat R;
      cv::Rodrigues(m, R);
      Eigen::Matrix3d ER;
      cv::cv2eigen(R, ER);
      return Eigen::Quaterniond(ER);
   }

   static void decode_quaternion(const Eigen::Quaterniond qin, Eigen::Vector3d& axis, double& angle)
   //-------------------------------------------------------------------------------------
   {
      Eigen::Quaterniond Q = qin;
      if (Q.w() > 1)
         Q.normalize(); // if w>1 acos and sqrt will produce errors, this cant happen if quaternion is normalised
      angle = 2 * acos(Q.w());
      double s = sqrt(1 - Q.w()*Q.w()); // assuming quaternion normalised then w is less than 1, so term always positive.
      if (s >= 0.001)
      { // test to avoid divide by zero, s is always positive due to sqrt
         // if s close to zero then direction of axis not important
         axis[0] = Q.x(); // if it is important that axis is normalised then replace with x=1; y=z=0;
         axis[1] = Q.y();
         axis[2] = Q.z();
      }
      else
      {
         axis[0] = Q.x() / s;
         axis[1] = Q.y() / s;
         axis[2] = Q.z() / s;
         axis.normalize();
      }
   }

   static void essential_decompose(const Eigen::Matrix3d& essential_matrix, Eigen::Matrix3d* rotation1,
                            Eigen::Matrix3d* rotation2, Eigen::Vector3d* translation)
   //------------------------------------------------------------------------------------------
   {
      Eigen::Matrix3d d;
      d << 0, 1, 0, -1, 0, 0, 0, 0, 1;

      const Eigen::JacobiSVD<Eigen::Matrix3d> svd(
            essential_matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
      Eigen::Matrix3d U = svd.matrixU();
      Eigen::Matrix3d V = svd.matrixV();
      if (U.determinant() < 0) {
         U.col(2) *= -1.0;
      }

      if (V.determinant() < 0) {
         V.col(2) *= -1.0;
      }

      // Possible configurations.
      *rotation1 = U * d * V.transpose();
      *rotation2 = U * d.transpose() * V.transpose();
      *translation = U.col(2).normalized();
   }


   inline double essential_check(const Eigen::Matrix3d R, const double tx, const double ty, const double tz,
                                 const double xt, const double yt, const double xq,  const double yq)
   {
      return  -R(0, 0)*ty*xt - R(0, 1)*ty*yt - R(0, 2)*ty + R(1, 0)*tx*xt + R(1, 1)*tx*yt +
               R(1, 2)*tx + xq*(-R(1, 0)*tz*xt - R(1, 1)*tz*yt - R(1, 2)*tz + R(2, 0)*ty*xt +
               R(2, 1)*ty*yt + R(2, 2)*ty) - yq*(-R(0, 0)*tz*xt - R(0, 1)*tz*yt - R(0, 2)*tz +
               R(2, 0)*tx*xt + R(2, 1)*tx*yt + R(2, 2)*tx);
   }

   static bool check_essential(Eigen::Matrix3d R, Eigen::Vector3d T, double epsilon =0.0000001)
   //------------------------------------------------------------------------------------
   {
      Eigen::Matrix3d E = skew33d(T)*R;
//      std::cout << R << std::endl << E << std::endl;
      if (! near_zero(E.determinant(), epsilon))
         return false;
      Eigen::Matrix3d EET = E*E.transpose();
      Eigen::Matrix3d constraint = 2*EET*E - EET.trace()*E;
//      std::cout << constraint << std::endl;
      struct ZeroVisit
      {
         double total = 0;
         void init(const double& value, Eigen::Index i, Eigen::Index j) {}
         void operator() (const double& value, Eigen::Index i, Eigen::Index j) { total += value; }
      } visitor;
      constraint.visit(visitor);
      if (! near_zero(visitor.total, epsilon))
         return false;

      Eigen::JacobiSVD<Eigen::MatrixXd, Eigen::FullPivHouseholderQRPreconditioner>
            svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
      auto U = svd.matrixU();
      auto V = svd.matrixV();
      auto S = svd.singularValues().asDiagonal().toDenseMatrix();
//      std::cout << "U: " << U << std::endl << "S: " << S << std::endl << "V: " << V << std::endl;
      if (!near_zero(S(2, 2), epsilon))
         return false;
   //   if ( (!near_zero(S(0, 0) - 1, 0.0000001)) || (!near_zero(S(1, 1) - 1, 0.0000001)) )
   //   {
   //      S(0, 0) = S(1, 1) = 1.0;
   //      S(2, 2) = 0.0;
   //      auto A = U*S*V.transpose();
   //      std::cout << A << std::endl;
   //   }
      return true;
   }

   // or just R = Eigen::Quaterniond::UnitRandom().toRotationMatrix();
   static void randomRotationMatrix(std::mt19937* generator, std::uniform_real_distribution<double>* distribution,
                                    Eigen::Matrix3d& R)
//-------------------------------------------------------------------------------------------------------------
   {
      double x = (*distribution)(*generator);
      double y = (*distribution)(*generator);
      double z = (*distribution)(*generator);
      double theta = std::acos(2*x - 1);
      double phi = 2*mut::PI*y;

      double r  = sqrt(z);
      double Vx = sin(phi) * r;
      double Vy = cos(phi) * r;
      double Vz = sqrt(2.f - z);

      double st = sin(theta);
      double ct = cos(theta);
      double Sx = Vx * ct - Vy * st;
      double Sy = Vx * st + Vy * ct;

      R(0,0) = Vx * Sx - ct;
      R(0,1) = Vx * Sy - st;
      R(0,2) = Vx * Vz;
      R(1,0) = Vy * Sx + st;
      R(1,1) = Vy * Sy - ct;
      R(1,2) = Vy * Vz;
      R(2,0) = Vz * Sx;
      R(2,1) = Vz * Sy;
      R(2,2) = 1.0 - z;

      assert(mut::near_zero(R.determinant() - 1.0, 0.000001));
   }

   static void random_vector3d_fixed_norm_kludge(const double norm_sqrt, Eigen::Vector3d& v)
   //-------------------------------------------------------------------------------
   {
      std::random_device rand_device;
      std::mt19937 rand_generator;
      std::unique_ptr<std::uniform_real_distribution<double>> distribution;
      std::uniform_int_distribution<int> index_distr(0,2);
      std::vector<double> pot;

      distribution.reset(new std::uniform_real_distribution<double>(0, norm_sqrt));
      double one = ((*distribution)(rand_generator));
      distribution.reset(new std::uniform_real_distribution<double>(0, std::max(norm_sqrt - one, 0.0)));
      double two = ((*distribution)(rand_generator));
      double three = std::max(norm_sqrt - one - two, 0.0);
      pot.clear();
      pot.push_back(one); pot.push_back(two); pot.push_back(three);
      int i1 = index_distr(rand_generator), i2 = index_distr(rand_generator), i3;
      while (i2 == i1) i2 = index_distr(rand_generator);
      switch (i1)
      {
         case 0: if (i2 == 1) i3 = 2; else i3 = 1; break;
         case 1: if (i2 == 0) i3 = 2; else i3 = 0; break;
         case 2: if (i2 == 0) i3 = 1; else i3 = 0; break;
      }
      v[0] = pot[i1]; v[1] = pot[i2]; v[2] = pot[i3];
   }

/*
int essential_best_pose(const Eigen::Matrix3d& essential_matrix,
                        const std::vector<FeatureCorrespondence>& normalized_correspondences,
                        Eigen::Matrix3d* rotation, Eigen::Vector3d* position)
{
   Eigen::Matrix3d rotation1, rotation2;
   Eigen::Vector3d translation;
   essential_decompose(essential_matrix, &rotation1, &rotation2, &translation);
   const std::vector<Eigen::Matrix3d> rotations = {rotation1, rotation1, rotation2, rotation2};
   const std::vector<Eigen::Vector3d> positions = {
         -rotations[0].transpose() * translation,
         -rotations[1].transpose() * -translation,
         -rotations[2].transpose() * translation,
         -rotations[3].transpose() * -translation};

   // From the 4 candidate poses, find the one with the most triangulated points
   // in front of the camera.
   std::vector<int> points_in_front_of_cameras(4, 0);
   for (int i = 0; i < 4; i++) {
      for (const auto& correspondence : normalized_correspondences) {
         if (IsTriangulatedPointInFrontOfCameras(
               correspondence, rotations[i], positions[i])) {
            ++points_in_front_of_cameras[i];
         }
      }
   }

   // Find the pose with the most points in front of the camera.
   const auto& max_element = std::max_element(points_in_front_of_cameras.begin(),
                                              points_in_front_of_cameras.end());
   const int max_index =
         std::distance(points_in_front_of_cameras.begin(), max_element);

   // Set the pose.
   *rotation = rotations[max_index];
   *position = positions[max_index];
   return *max_element;
}
*/
}
#endif
