#ifndef __KEYPOINTFLANN_HH
#define __KEYPOINTFLANN_HH

#include <vector>

#include <opencv2/core/core.hpp>

#include "nanoflann.hpp"

struct KeyPointFlannSource
//==========================
{
   std::vector<cv::KeyPoint>* ppoints;

   void set(std::vector<cv::KeyPoint>* pts) { ppoints = pts; }

   cv::KeyPoint& get(size_t i) { return (*ppoints)[i]; }

   bool is_selected = false;

   inline size_t kdtree_get_point_count() const { return ppoints->size(); }

   inline float kdtree_get_pt(const size_t i, int dim) const
   //-----------------------------------------------------
   {
      if (dim == 0) return (*ppoints)[i].pt.x;
      else if (dim == 1) return (*ppoints)[i].pt.y;
      else throw std::logic_error("Invalid dimension in kd-tree for ImageWindow");
   }

   template <class BBOX>
   bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }
};

typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, KeyPointFlannSource>,
      KeyPointFlannSource, 2, size_t> keypoint_kd_tree_t;

struct ThreeDFlannSource
//==========================
{
   std::vector<std::tuple<cv::Point3d, cv::KeyPoint, cv::Mat>>* ppoints;

   void set(std::vector<std::tuple<cv::Point3d, cv::KeyPoint, cv::Mat>>* pts) { ppoints = pts; }

   std::tuple<cv::Point3d, cv::KeyPoint, cv::Mat>& get(size_t i) { return (*ppoints)[i]; }

   bool is_selected = false;

   inline size_t kdtree_get_point_count() const { return ppoints->size(); }

   inline float kdtree_get_pt(const size_t i, int dim) const
   //-----------------------------------------------------
   {
//      cv::KeyPoint kp = std::get<1>(t)
      const std::tuple<cv::Point3d, cv::KeyPoint, cv::Mat>& tt = (*ppoints)[i];
      const cv::KeyPoint& kp = std::get<1>(tt);
      if (dim == 0) return kp.pt.x;
      else if (dim == 1) return kp.pt.y;
      else throw std::logic_error("Invalid dimension in kd-tree ThreeDFlannSource");
   }

   template <class BBOX>
   bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }
};

typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, ThreeDFlannSource>,
      ThreeDFlannSource, 2, size_t> threeD_kd_tree_t;


#endif //__KEYPOINTFLANN_HH
