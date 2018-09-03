
#include "ocv.h"

#include <iostream>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#ifdef HAVE_OPENCV_XFEATURES2D
#include <opencv2/xfeatures2d.hpp>
#endif

#include "KeypointFlann.hh"

namespace ocv
{
   bool detect(cv::Ptr<cv::Feature2D> detector, cv::Mat &image, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors,
               int top_n)
//--------------------------------------------------------------------------------------------------
   {
      keypoints.clear();
      descriptors.release();
      std::vector<cv::KeyPoint> initial_keypoints;
      try
      {
         if (top_n < 0) top_n = 1000;
         detector->detect(image, initial_keypoints);
         cv::KeyPointsFilter::removeDuplicated(initial_keypoints);
         cv::KeyPointsFilter::retainBest(initial_keypoints, top_n);
         if (!initial_keypoints.empty())
         {
            std::vector<bool> initial_keypoint_duplicates(initial_keypoints.size(), false);
            KeyPointFlannSource index_points;
            index_points.set(&initial_keypoints);
            keypoint_kd_tree_t index(2, index_points, nanoflann::KDTreeSingleIndexAdaptorParams(10));
            index.buildIndex();
            nanoflann::SearchParams params;
            std::vector<std::pair<size_t, float>> radius_matches;
            for (cv::KeyPoint &kp : initial_keypoints)
            {
               const cv::Point2f& pt = kp.pt;
               const float query_pt[2] = { pt.x, pt.y };
               const size_t no = index.radiusSearch(&query_pt[0], 1.2, radius_matches, params);
               if (no > 1)
               {
                  bool is_duplicate = false;
                  for (std::pair<size_t, float> rmatch : radius_matches)
                     if ( (is_duplicate = initial_keypoint_duplicates[rmatch.first]) )
                        break;
                  if (is_duplicate)
                     continue;
                  else
                     initial_keypoint_duplicates[radius_matches[0].first] = true;
               }
               if (detector->getDefaultName() != "Feature2D.AKAZE") // AKAZE compute uses the KeyPoint class_id
               {
                  KeyPointExtra_t kpe;
                  kpe.class_id = static_cast<unsigned int>(kp.class_id);
                  kpe.is_selected = 0;
                  kp.class_id = kpe.class_id;
               }
               keypoints.push_back(kp);
            }
            detector->compute(image, keypoints, descriptors);
            if (detector->getDefaultName() == "Feature2D.AKAZE")
            {
               for (cv::KeyPoint kp : keypoints)
               {
                  KeyPointExtra_t kpe;
                  kpe.class_id = static_cast<unsigned int>(kp.class_id);
                  kpe.is_selected = 0;
                  kp.class_id = kpe.class_id;
               }
            }
         }
      }
      catch (std::exception &e)
      {
         std::cerr << "ERROR: detect: " << e.what() << std::endl;
         return false;
      }
      return (initial_keypoints.size() > 0);
   }

   bool match(cv::Ptr<cv::DescriptorMatcher>& matcher,
              std::vector<cv::KeyPoint>& train_keypoints, std::vector<cv::KeyPoint>& query_keypoints,
              cv::Mat& train_descriptors, cv::Mat& query_descriptors,
              std::vector<cv::KeyPoint>& matched_train_keypoints, std::vector<cv::KeyPoint>& matched_query_keypoints,
              cv::Mat& matched_train_descriptors, cv::Mat& matched_query_descriptors,
              std::vector<cv::DMatch> &matches, std::vector<std::pair<cv::Point3d, cv::Point3d>>* matched_points,
              std::stringstream* errs)
//------------------------------------------------------------------------------------------------------------------------
   {
      std::vector<std::vector<cv::DMatch>> knnMatches;
      matched_train_keypoints.clear(); matched_query_keypoints.clear();
      matched_train_descriptors.resize(0); matched_query_descriptors.resize(0);
      matches.clear();
      try
      {
         matcher->knnMatch(query_descriptors, train_descriptors, knnMatches, 2);
         const float minRatio = 0.75f;
         for (size_t i = 0; i < knnMatches.size(); i++)
         {
            const cv::DMatch &bestMatch = knnMatches[i][0];
            const cv::DMatch &betterMatch = knnMatches[i][1];

            // To avoid NaN's when match has zero distance use inverse ratio.
            float inverseRatio = bestMatch.distance / betterMatch.distance;

            // Test for distinctiveness: pass only matches where the inverse
            // ratio of the distance between nearest matches is < than minimum.
            if (inverseRatio < minRatio)
            {
               cv::KeyPoint &tkp = train_keypoints[bestMatch.trainIdx];
               cv::KeyPoint &qkp = query_keypoints[bestMatch.queryIdx];
               matched_train_keypoints.push_back(tkp);
               matched_query_keypoints.push_back(qkp);
               matched_train_descriptors.push_back(train_descriptors.row(bestMatch.trainIdx));
               matched_query_descriptors.push_back(query_descriptors.row(bestMatch.queryIdx));
//            std::vector<cv::DMatch> vd;
//            vd.push_back(bestMatch);
//            vd.push_back(betterMatch);
               cv::DMatch m(static_cast<int>(matched_train_keypoints.size() - 1),
                            static_cast<int>(matched_query_keypoints.size() - 1),
                            bestMatch.distance);
               matches.push_back(m);
               if (matched_points != nullptr)
               {
                  cv::Point2f& tpt = tkp.pt;
                  cv::Point2f& qpt = qkp.pt;
                  matched_points->emplace_back(std::make_pair(cv::Point3d(tpt.x, tpt.y, 1), cv::Point3d(qpt.x, qpt.y, 1)));
               }
            }
         }
      }
      catch (std::exception &e)
      {
         if (errs != nullptr)
            *errs << "ERROR: match " << e.what();
         std::cerr << "ERROR: match " << e.what() << std::endl;
         return false;
      }
      return (matches.size() > 0);
   }

   bool homography(const std::vector<cv::KeyPoint>& train_keypoints, const std::vector<cv::KeyPoint>& query_keypoints,
                   const std::vector<cv::DMatch>& matches, cv::Mat& homography, std::vector<cv::DMatch>& homography_matches,
                   std::vector<std::pair<cv::Point3d, cv::Point3d>>* matched_points, int method, float reprojection_error)
//------------------------------------------------------------------------------------------
   {
      try
      {
         size_t n = static_cast<int>(matches.size());
         std::vector<unsigned char> inliers(n);
         std::vector<cv::Point2f> train_points(n), query_points(n);
         for (size_t i = 0; i < matches.size(); i++)
         {
            train_points[i] = train_keypoints[matches[i].trainIdx].pt;
            query_points[i] = query_keypoints[matches[i].queryIdx].pt;
         }

         homography = cv::findHomography(train_points, query_points, method, reprojection_error, inliers);
         std::vector<cv::Point2f> homography_pts;
         for (size_t i = 0; i < inliers.size(); i++)
         {
            if (inliers[i])
            {
               const cv::DMatch& m = matches[i];
               homography_matches.emplace_back(m.queryIdx, m.trainIdx, m.imgIdx, m.distance);
               if (matched_points != nullptr)
               {
                  const cv::Point2f& tpt = train_points[i];
                  const cv::Point2f& qpt = query_points[i];
                  matched_points->emplace_back(std::make_pair(cv::Point3d(tpt.x, tpt.y, 1), cv::Point3d(qpt.x, qpt.y, 1)));
               }
            }
         }
      }
      catch (std::exception &e)
      {
         std::cerr << "FeatureBasedDetector::find_homography Exception: " << e.what() << std::endl;
         return false;
      }
      return (homography_matches.size() > MIN_HOMOGRAPHY_RESULT);
   }

   bool convex_hull(std::vector<cv::KeyPoint>& keypoints,// const cv::Mat& descriptors,
                    std::vector <cv::Point2f> &hull_points, cv::Point2f &centroid,
                    cv::RotatedRect& rbb, cv::Rect2f& bb,
                    std::vector<cv::KeyPoint>* hull_keypoints, //cv::Mat* hull_descriptors,
                    std::stringstream *errs)
//-------------------------------------------------------------------------------------------------------------------
   {
      if (keypoints.size() < 3)
      {
         if (errs)
            *errs << "Not enough points to find convex hull";
         return false;
      }
      std::vector<cv::Point2f> points;
      hull_points.clear();
      cv::KeyPoint::convert(keypoints, points);

      cv::convexHull(points, hull_points);
      size_t n = hull_points.size();
      if (n < 3)
      {
         if (errs)
            *errs << "Convex hull less than 3 points";
         return false;
      }
      centroid.x = centroid.y = 0;
      for (size_t i = 0; i < n; i++)
      {
         const cv::Point2f& pt = hull_points[i];
         centroid.x += pt.x;
         centroid.y += pt.y;
      }
      centroid.x /= static_cast<float>(n);
      centroid.y /= static_cast<float>(n);
      if (n >= 4)
      {
         try
         {
            rbb = cv::minAreaRect(hull_points);
            bb = rbb.boundingRect2f();
         }
         catch (cv::Exception& cverr)
         {
            rbb = cv::RotatedRect();
            bb = cv::Rect2f(0, 0, 0, 0);
            if (errs)
               *errs << "WARNING: Finding bounding rectangles gave " << cverr.what();
            std::cerr <<  "ocv::convex_hull - WARNING: Finding bounding rectangles gave " << cverr.what();
         }
      }
      else
      {
         rbb = cv::RotatedRect();
         bb = cv::Rect2f(0, 0, 0, 0);
         if (errs)
            *errs << "WARNING: Convex hull less than 4 points. Bounding rectangles cannot be found";
      }
      return true;
   }

   cv::Mat& draw_matches(const cv::Mat& train_image, const std::vector<cv::KeyPoint>& train_keypoints,
                         const cv::Mat& query_image, const std::vector<cv::KeyPoint>& query_keypoints,
                         const std::vector<cv::DMatch>& matches, cv::Mat& match_img, std::string debug_savename,
                         cv::Scalar pointColor, cv::Scalar lineColor)
//----------------------------------------------------------------------------------------------------------------------
   {
      cv::drawMatches(train_image, train_keypoints, query_image, query_keypoints,
                      matches, match_img, lineColor, pointColor);
      if (! debug_savename.empty())
         cv::imwrite(debug_savename, match_img);
      return match_img;
   }

   void convert2x3(const std::vector<cv::KeyPoint>& keypoints, std::vector<cv::Point3d>& points, bool isPoint)
   //--------------------------------------------------------------------------------------------
   {
      points.clear();
      for (cv::KeyPoint kpt : keypoints)
      {
         cv::Point2f& pt = kpt.pt;
         points.emplace_back(pt.x, pt.y, ((isPoint) ? 1 : 0));
      }
   }

   void matched_pairs(const std::vector<cv::KeyPoint>& keypoints, const std::vector<cv::DMatch>& matches,
                      std::vector<std::pair<cv::Point3d, cv::Point3d>>& pts)
   //-------------------------------------------------------------------------------------------------
   {
      pts.clear();
   }


   void plot_circles(const cv::Mat& img, double x, double y, const std::vector<cv::Scalar>& colors)
//----------------------------------------------------------------------------------------------------------------
   {
      int i = 2;
      for (cv::Scalar color : colors)
      {
         cv::circle(img, cv::Point2i(cvRound(x), cvRound(y)), i, color, 1);
         i++;
      }
   }

   void plot_rectangles(const cv::Mat& img, double x, double y, const std::vector<cv::Scalar>& colors)
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

   void reprojection3d(const Eigen::Matrix3d& K, const Eigen::Matrix3d& R, const Eigen::Vector3d& T,
                       const std::vector<cv::Point3d>& world_pts, const std::vector<cv::Point2d>& image_pts,
                       double& max_error, double &mean_error,
                       const cv::Mat* img, cv::Mat* project_image, bool draw_query,
                       cv::Scalar query_color, cv::Scalar projection_color)
   //--------------------------------------------------------------------------------------------------------------------
   {
      if (world_pts.empty()) return;
      max_error = std::numeric_limits<double >::min(); mean_error = 0;
      std::vector<cv::Scalar> projectionColor(7), queryColor(7);
      bool is_draw = ( (img != nullptr) && (! img->empty()) && (project_image != nullptr) );
      if (is_draw)
      {
         img->copyTo(*project_image);
         std::fill(projectionColor.begin(), projectionColor.begin() + 7, projection_color);
         std::fill(queryColor.begin(), queryColor.begin() + 7, query_color);
      }
      std::vector<double> errors;
      for (size_t i=0; i<world_pts.size(); i++)
      {
         cv::Point3d wpt = world_pts[i];
         Eigen::Vector3d ept;
         Eigen::Vector3d p = K*(R*Eigen::Vector3d(wpt.x, wpt.y, wpt.z) + T);
         p /= p[2];
         if (is_draw)
            plot_circles(*project_image, p[0], p[1], projectionColor);
         if (image_pts.size() > i)
         {
            const cv::Point2d& qpt = image_pts[i];
            if (is_draw)
               plot_rectangles(*project_image, qpt.x, qpt.y, queryColor);
            double error = cv::norm(cv::Vec2d(p[0], p[1]) - cv::Vec2d(qpt.x, qpt.y));
            if (error > max_error)
               max_error = error;
            mean_error += error;
            errors.push_back(error);
         }
      }
      mean_error /= world_pts.size();
   }

   void reprojection2d(const Eigen::Matrix3d& K, const Eigen::Matrix3d& R, const Eigen::Vector3d& T,
                       const std::vector<cv::Point3d>& train_pts, const std::vector<cv::Point3d>& query_pts,
                       const double depth,
                       double& max_error, double& mean_error,
                       const cv::Mat* img, cv::Mat* project_image, bool draw_query, cv::Scalar query_color,
                       cv::Scalar projection_color)
//----------------------------------------------------------------------------------------------------------------
   {
      if (train_pts.empty()) return;
      max_error = std::numeric_limits<double >::min(); mean_error = 0;
      std::vector<cv::Scalar> projectionColor(7), queryColor(7);
      if ( (img  != nullptr) && (project_image  != nullptr) )
      {
         img->copyTo(*project_image);
         std::fill(projectionColor.begin(), projectionColor.begin() + 7, projection_color);
         std::fill(queryColor.begin(), queryColor.begin() + 7, query_color);
      }
      Eigen::Matrix3d KI = K.inverse();
      for (size_t j=0; j<train_pts.size(); j++)
      {
         const cv::Point3d& tpt = train_pts[j];
         Eigen::Vector3d ray = KI*Eigen::Vector3d(tpt.x, tpt.y, 1);
         ray /= ray[2];
         if (! std::isnan(depth))
            ray *= depth;
         Eigen::Vector3d ipt = R*ray + T;
//      std::cout << (R*ray).transpose() << " + " << T.transpose() << " = " << ipt.transpose() << std::endl;
         ipt = K*ipt;
         ipt /= ipt[2];
         if (query_pts.size() > j)
         {
            const cv::Point3d& qpt = query_pts[j];
            if ((project_image != nullptr) && (!project_image->empty()))
            {
               if (draw_query)
                  plot_rectangles(*project_image, qpt.x, qpt.y, queryColor);
               plot_circles(*project_image, ipt[0], ipt[1], projectionColor);
            }

            double error = (ipt - Eigen::Vector3d(qpt.x, qpt.y, qpt.z)).norm();
            if (error > max_error)
               max_error = error;
            mean_error += error;
         }
//     std::cout << j << ": " << qpt << " " << ipt.transpose() << " = " << error << std::endl;
      }
      mean_error /= train_pts.size();
   }

   void cvLabel(const cv::Mat& img, const std::string label, const cv::Point &pt, const cv::Scalar background,
                const cv::Scalar foreground, const int font)
//--------------------------------------------------------------------------------------------------------
   {
      double scale = 0.4;
      int thickness = 1;
      int baseline = 0;
      cv::Size text = cv::getTextSize(label, font, scale, thickness, &baseline);
      cv::rectangle(img, pt + cv::Point(0, baseline), pt + cv::Point(text.width, -text.height), background, CV_FILLED);
      cv::putText(img, label, pt, font, scale, foreground, thickness, 8);
   }

}