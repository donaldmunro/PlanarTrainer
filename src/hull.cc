#include <iostream>
#include <unordered_map>

#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>

#include "hull.h"
#include "typ.h"

typedef struct
   {
      std::size_t operator()(const cv::Point2f& pt) const
      //---------------------------------------------
      {
         size_t h = 2166136261U;
         const size_t scale = 16777619U;
         Cv32suf u;
         float x = pt.x;
         float y = pt.y;
         u.f = x; h = (scale * h) ^ u.u;
         u.f = y; h = (scale * h) ^ u.u;
         return h;
      }
   } PointHash;

   typedef struct
   {
      bool operator()(const cv::Point2f& pt1, const cv::Point2f& pt2) const
      //------------------------------------------------------------
      {
         PointHash hash;
         return (hash(pt1) == hash(pt2));
      }
   } PointEq;

int get_convex_hull(std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors, std::vector <cv::Point2f> &hull_points,
                    std::vector<cv::KeyPoint>& hull_keypoints, cv::Mat &hull_descriptors, cv::Point2f &centroid,
                    bool is_ref)
//-------------------------------------------------------------------------------------------------------------------
{
   std::vector<cv::Point2f> points;
   hull_points.clear();
   std::unordered_map<cv::Point2f, int, PointHash, PointEq> pt_lookup;
   for (int i=0; i< static_cast<int>(keypoints.size()); i++)
   {
      const cv::KeyPoint& kp = keypoints[i];
      KeyPointExtra_t kpc;
      kpc.class_id = static_cast<unsigned int>(kp.class_id);
      if (is_ref)
         kpc.is_hull = 0;
      if (kpc.is_selected == 1)
      {
         const cv::Point2f& pt = kp.pt;

         auto it = pt_lookup.find(pt);
         if (it != pt_lookup.end())
            std::cerr << "Duplicate convex hull point (" << pt.x << ", " << pt.y << ")" << std::endl;
         else
         {
            points.push_back(pt);
            pt_lookup[pt] = i;
         }
      }
   }
   if (points.size() > 0)
      cv::convexHull(points, hull_points);
   const int n = static_cast<int>(hull_points.size());
   if (n > 0)
   {
      centroid.x = centroid.y = 0;
      for (int i = 0; i < n; i++)
      {
         cv::Point2f pt = hull_points[i];
         auto it = pt_lookup.find(pt);
         if (it != pt_lookup.end())
         {
            centroid.x += pt.x;
            centroid.y += pt.y;
            int j = it->second;
            cv::KeyPoint lkp(keypoints[j]);
            cv::KeyPoint& kp = (is_ref) ? keypoints[j] : lkp;
            KeyPointExtra_t kpe;
            kpe.class_id = static_cast<uint32_t>(kp.class_id);
            kpe.is_hull = 1;
            kp.class_id = kpe.class_id;
            hull_keypoints.push_back(kp);
            cv::Mat row = descriptors.row(j);
            hull_descriptors.push_back(row);
//               cv::Mat row(1, descriptors.cols, descriptors.type());
//               for (int k=0; k<descriptors.cols; k++)
//                  row.at<float>(0, k) = (descriptors.at<float>(j, k));
//               hull_descriptors.push_back(row);


         }
         else
         {
            std::cerr << "Point (" << pt.x << ", " << pt.y << ") not found in KeyPoint map " << std::endl;

         }
      }
      centroid.x /= static_cast<float>(n);
      centroid.y /= static_cast<float>(n);
   }
   return n;
}

void show_convex_hull(const std::string win_name, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
//-----------------------------------------------------------------------------------------------------------
{
   std::vector<cv::Point2f> hull_points;
   std::vector<cv::KeyPoint> hull_keypoints;
   cv::Mat hull_descriptors, display;
   cv::Point2f centroid;
   cv::Mat img;
   int n = get_convex_hull(keypoints, descriptors, hull_points, hull_keypoints, hull_descriptors, centroid, false);
   if (n > 0)
   {
      cv::drawKeypoints(img, hull_keypoints, display, cv::Scalar(0,0,255));
      cv::Point pt0 = hull_points[n - 1];
      for (int i = 0; i < n; i++)
      {
         cv::Point pt = hull_points[i];
         line(display, pt0, pt, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
         pt0 = pt;
      }
      cv::imshow("hull", display);
   }
   cv::namedWindow(win_name, CV_WINDOW_AUTOSIZE);
   cv::moveWindow(win_name, 0, img.rows+3);
   cv::imshow(win_name, display);
//   cv::Ptr<std::vector<cv::KeyPoint>> selected_keypoints(new std::vector<cv::KeyPoint>(hull_keypoints));
//   cv::Ptr<cv::Mat> selected_descriptors(new cv::Mat(hull_descriptors));
//   show_homography(selected_keypoints, selected_descriptors);
}