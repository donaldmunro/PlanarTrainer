#include "PoseRANSAC.hh"

inline Eigen::Matrix3d skew33d(Eigen::Vector3d v3)
//-----------------------------------------------
{
   double a = v3[0], b = v3[1], c = v3[2];
   Eigen::Matrix3d X;
   X << 0, -c, b, c, 0, -a, -b, a, 0;
   return X;
}

#ifdef USE_THEIA_RANSAC
bool Grav2DRansacEstimator::EstimateModel(const std::vector<std::pair<cv::Point3d, cv::Point3d>>& matches,
                                          std::vector<GravPoseRansacModel>* models) const
//------------------------------------------------------------------------------------------
{
   Eigen::Vector3d T;
   std::vector<cv::Point3d> train_pts, query_pts;
   std::for_each (std::begin(matches), std::end(matches),
   [&train_pts, &query_pts](const std::pair<cv::Point3d, cv::Point3d>& pp)
         //--------------------------------------------------------
   {
      train_pts.push_back(pp.first);
      query_pts.push_back(pp.second);
   });

   if ( (std::isnan(depth)) || (depth <= 0) )
      pose2d::pose_translation(KI, train_pts, query_pts, R, T);
   else
      pose2d::pose_translation(KI, train_pts, query_pts, depth, R, T);
   if ( (! std::isnan(T[0])) && (! std::isnan(T[1])) && (! std::isnan(T[2])) )
      models->emplace_back(Q, T);
   return true;
}

double Grav2DRansacEstimator::Error(const std::pair<cv::Point3d, cv::Point3d> &match, const GravPoseRansacModel &model) const
//------------------------------------------------------------------------------------------------------------
{
//   double error = std::numeric_limits<double>::infinity();
   const Eigen::Vector3d& T = model.translation;
   if ( (std::isnan(depth)) || (depth <= 0) )
   {
      Eigen::Matrix3d E = skew33d(T) * R;
      const cv::Point3d& tpt = match.first;
      const cv::Point3d& qpt = match.second;
      Eigen::Vector3d Xt = KI*Eigen::Vector3d(tpt.x, tpt.y, tpt.z);
      Eigen::Vector3d Xq = KI*Eigen::Vector3d(qpt.x, qpt.y, qpt.z);
      double error = Xq.transpose() * E * Xt;
      return error;
   }
   else
   {
      const cv::Point3d tpt = match.first;
      Eigen::Vector3d train_ray = KI*Eigen::Vector3d(tpt.x, tpt.y, 1);
      train_ray /= train_ray[2];
      train_ray *= depth;
      Eigen::Vector3d ipt = EK*(R*train_ray + T);
      ipt /= ipt[2];
      const cv::Point3d pt = match.second;
      Eigen::Vector3d qpt(pt.x, pt.y, 1);
      Eigen::Vector3d residual = (ipt - qpt);
      return residual.dot(residual);
   }
//   std::cout << "Grav2DRansacEstimator::Error " << error << " " << T.transpose() << "\n";
//   return abs(error);
}

bool Grav3DRansacEstimator::EstimateModel(const std::vector<std::pair<cv::Point3d, cv::Point2d>> &matches,
                                          std::vector<GravPoseRansacModel> *models) const
//----------------------------------------------------------------------------------------------------
{
   Eigen::Vector3d T;
   std::vector<cv::Point3d> train_pts;
   std::vector<cv::Point2d> query_pts;
   std::for_each (std::begin(matches), std::end(matches),
   [&train_pts, &query_pts](const std::pair<cv::Point3d, cv::Point2d>& pp)
   //--------------------------------------------------------
   {
      const cv::Point3d& wpt = pp.first;
      const cv::Point2d& ipt = pp.second;
      train_pts.emplace_back(wpt.x, wpt.y, wpt.z);
      query_pts.emplace_back(ipt.x, ipt.y);
   });

   pose3d::pose_translation(train_pts, query_pts, KI, R, T);
   if ( (! std::isnan(T[0])) && (! std::isnan(T[1])) && (! std::isnan(T[2])) )
      models->emplace_back(Q, T);
   return true;
}

double Grav3DRansacEstimator::Error(const std::pair<cv::Point3d, cv::Point2d> &match, const GravPoseRansacModel &model) const
//-------------------------------------------------------------------------------------------------------------------------
{
   const cv::Point3d& wpt = match.first;
   const cv::Point2d& qpt = match.second;
   const double tx = model.translation[0], ty = model.translation[1], tz = model.translation[2];
   Eigen::Vector3d p = K * (R * Eigen::Vector3d(wpt.x, wpt.y, wpt.z) + Eigen::Vector3d(tx, ty, tz));
   p /= p[2];
   Eigen::Vector3d p2(qpt.x, qpt.y, 1);
   Eigen::Vector3d pp = p - p2;
   return pp.squaredNorm();
}
#else
const int Grav2DRansacEstimator::estimate(const Grav2DRansacData& samples, const std::vector<size_t>& sampleIndices,
                                          templransac::RANSACParams& parameters,
                                          std::vector<GravPoseRansacModel>& models) const

//------------------------------------------------------------------------------------------------------------------
{
   std::vector<cv::Point3d> train_img_pts;
   std::vector<cv::Point3d> query_image_pts;
   RANSAC_copy_points(samples, sampleIndices, train_img_pts, query_image_pts);
   Eigen::Vector3d T;
   pose2d::pose_translation(KI, train_img_pts, query_image_pts, R, T);
   if ( (! std::isnan(T[0])) && (! std::isnan(T[1])) && (! std::isnan(T[2])) )
      models.emplace_back(Q, T);
   return models.size();
}

const void Grav2DRansacEstimator::error(const Grav2DRansacData &samples, const std::vector<size_t> &sampleIndices,
                                        GravPoseRansacModel &model, std::vector<size_t> &inlier_indices,
                                        std::vector<size_t> &outlier_indices, double error_threshold) const
//---------------------------------------------------------------------------------------------------------------
{
   std::vector<cv::Point3d> train_img_pts;
   std::vector<cv::Point3d> query_img_pts;
   size_t no = RANSAC_copy_points(samples, sampleIndices, train_img_pts, query_img_pts);
   Eigen::Vector3d& T = model.translation;
//   Eigen::Matrix<double, 3, 4> A;
//   A.block(0, 0, 3, 3) = R;
//   A(0, 3) = T[0]; A(1, 3) = T[1]; A(2, 3) = T[2];
   Eigen::Matrix3d E = skew33d(T) * R;
   for (size_t i=0; i<no; i++)
   {
//      Eigen::Vector3d train_ray = KI*train_img_pts[i];
//      Eigen::Vector3d query_ray = KI*query_img_pts[i];
//      Eigen::Vector3d tpv(train_ray[0], train_ray[1], 1);
//      Eigen::Vector3d qpv(query_ray[0], query_ray[1], 1);
//      Eigen::Matrix<double, 3, 1> Ax = A*Eigen::Matrix<double, 4, 1>(tpv[0], tpv[1], tpv[2], 1);
//      Ax /= Ax[2];
//      Eigen::Matrix<double, 3, 1> errvec = Ax - qpv;
//      double error = errvec.block(0,0,2,1).norm();
      const cv::Point3d& tpt = train_img_pts[i];;
      const cv::Point3d& qpt = query_img_pts[i];
      Eigen::Vector3d Xt = KI*Eigen::Vector3d(tpt.x, tpt.y, tpt.z);
      Eigen::Vector3d Xq = KI*Eigen::Vector3d(qpt.x, qpt.y, qpt.z);
      double error = Xq.transpose() * E * Xt;
      if (error < error_threshold)
         inlier_indices.push_back(i);
      else
         outlier_indices.push_back(i);
   }
}

const int Grav2DDepthRansacEstimator::estimate(const Grav2DRansacData& samples, const std::vector<size_t>& sampleIndices,
                                               templransac::RANSACParams& parameters,
                                                std::vector<GravPoseRansacModel>& models) const
//-----------------------------------------------------------------------------------------------------------
{
   std::vector<cv::Point3d> train_img_pts;
   std::vector<cv::Point3d> query_image_pts;
   RANSAC_copy_points(samples, sampleIndices, train_img_pts, query_image_pts);
   Eigen::Vector3d T;
   pose2d::pose_translation(KI, train_img_pts, query_image_pts, depth, R, T);
   if ( (! std::isnan(T[0])) && (! std::isnan(T[1])) && (! std::isnan(T[2])) )
      models.emplace_back(Q, T);
   return models.size();
}

const void Grav2DDepthRansacEstimator::error(const Grav2DRansacData& samples, const std::vector<size_t>& sampleIndices,
                                             GravPoseRansacModel& model, std::vector<size_t>& inlier_indices,
                                             std::vector<size_t>& outlier_indices, double error_threshold) const
//---------------------------------------------------------------------------------------------------------
{
   std::vector<cv::Point3f> train_img_pts;
   std::vector<cv::Point3f> query_img_pts;
   size_t no = RANSAC_copy_points(samples, sampleIndices, train_img_pts, query_img_pts);
   Eigen::Vector3d& T = model.translation;
//   Eigen::Matrix<double, 3, 4> A;
//   A.block(0, 0, 3, 3) = R;
//   A(0, 3) = T[0]; A(1, 3) = T[1]; A(2, 3) = T[2];
   double err = error_threshold*error_threshold;
   for (size_t i=0; i<no; i++)
   {
      const cv::Point3d tpt = train_img_pts[i];
      Eigen::Vector3d train_ray = KI*Eigen::Vector3d(tpt.x, tpt.y, 1);
      train_ray /= train_ray[2];
      train_ray *= depth;
      Eigen::Vector3d ipt = EK*(R*train_ray + T);
      ipt /= ipt[2];
      const cv::Point3d pt = query_img_pts[i];
      Eigen::Vector3d qpt(pt.x, pt.y, 1);
      Eigen::Vector3d residual = (ipt - qpt);
      double error =  residual.dot(residual);
      if (error < err)
         inlier_indices.push_back(i);
      else
         outlier_indices.push_back(i);

//      train_ray = KI*train_img_pts[i];
//      Eigen::Vector3d query_ray = KI*query_img_pts[i];
//      Eigen::Vector3d tpv(train_ray[0], train_ray[1], 1);
//      Eigen::Vector3d qpv(query_ray[0], query_ray[1], 1);
//      Eigen::Matrix<double, 3, 1> Ax = A*Eigen::Matrix<double, 4, 1>(tpv[0], tpv[1], tpv[2], 1);
//      Ax /= Ax[2];
//      Eigen::Matrix<double, 3, 1> errvec = Ax - qpv;
   }
}


const int Grav3DRansacEstimator::estimate(const Grav3DRansacData& samples, const std::vector<size_t>& sampleIndices,
                                          templransac::RANSACParams& parameters,
                                          std::vector<GravPoseRansacModel>& models) const
//------------------------------------------------------------------------------------------------------------------
{
   std::vector<cv::Point3d> world_pts;
   std::vector<cv::Point2d> query_image_pts;
   copy_points3d(samples, sampleIndices, world_pts, query_image_pts);
   Eigen::Vector3d T;
   pose3d::pose_translation(world_pts, query_image_pts, KI, R, T);
   if ( (! std::isnan(T[0])) && (! std::isnan(T[1])) && (! std::isnan(T[2])) )
      models.emplace_back(Q, T);
   return models.size();
}

const void Grav3DRansacEstimator::error(const Grav3DRansacData& samples, const std::vector<size_t>& sampleIndices,
                                        GravPoseRansacModel& model, std::vector<size_t>& inlier_indices,
                                        std::vector<size_t>& outlier_indices, double error_threshold) const
//-----------------------------------------------------------------------------------------------------------
{
   std::vector<cv::Point3d> world_pts;
   std::vector<cv::Point2d> query_image_pts;
   size_t no = copy_points3d(samples, sampleIndices, world_pts, query_image_pts);
   const double error_threshold_square = error_threshold * error_threshold, tx = model.translation[0],
                ty = model.translation[1], tz = model.translation[2];
#ifdef _RANSAC_STATS_
   double min_error = std::numeric_limits<double>::max(), max_error = std::numeric_limits<double >::min(), mean_error = 0;
#endif
   Eigen::Matrix3d R = Q.toRotationMatrix();
   for (size_t i = 0; i < no; i++)
   {
      cv::Point3d& wpt = world_pts[i];
      Eigen::Vector3d p = K * (R * Eigen::Vector3d(wpt.x, wpt.y, wpt.z) + Eigen::Vector3d(tx, ty, tz));
      p /= p[2];
      Eigen::Vector3d p2(query_image_pts[i].x, query_image_pts[i].y, 1);
      Eigen::Vector3d pp = p - p2;
      double error = pp.dot(pp);
      if (error < error_threshold_square)
         inlier_indices.push_back(i);
      else
         outlier_indices.push_back(i);
#ifdef _RANSAC_STATS_
      min_error = std::min(min_error, error);
      max_error = std::max(max_error, error);
      mean_error += error;
#endif
   }
#ifdef _RANSAC_STATS_
   mean_error /= no;
   std::cout << "Grav3DRansacEstimator Error^2: " << min_error << " " << max_error << " " << mean_error << " ("
             << inlier_indices.size() << "/" << outlier_indices.size() << ")" << std::endl;

#endif
}
#endif