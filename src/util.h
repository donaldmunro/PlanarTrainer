#ifndef _INC_UTIL_
#define _INC_UTIL_

#include <string>

#include <opencv2/core/core.hpp>

std::string type(cv::InputArray a);

std::string trim(std::string str, std::string chars = " \t");

std::size_t split(std::string s, std::vector<std::string>& tokens, std::string delim ="\t\n ");

inline float euclidean2(const cv::Point2f& pt1, const cv::Point2f& pt2)
//---------------------------------------------------------------------
{
   float dx = pt2.x - pt1.x;
   float dy = pt2.y - pt1.y;
   return dx*dx + dy*dy;
}

void shift(cv::Mat &img, int shift, bool is_right, cv::Mat &out);

template<typename T, int c>
inline void cvmatcol_to_cvvec(cv::Mat& m, int colno, cv::Vec<T, c>& v)
//--------------------------------------------------------------------
{
   const cv::Mat &mc = m.col(colno).t();
   const T* p = mc.ptr<T*>(0);
   for (int i=0; i<c; i++)
      v[i] = *p++;
}

template<typename T, int c>
inline void cvvec_col(cv::Mat& m, int colno, cv::Vec<T, c>& v)
//------------------------------------------------------------
{
   for (int i=0; i<c; i++)
      m.at<T>(i, colno) = v[i];
}

#endif