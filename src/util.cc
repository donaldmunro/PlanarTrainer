#include <string>

#include <opencv2/core/core.hpp>

#include "util.h"

std::string type(cv::InputArray a)
//--------------------------------
{
   int numImgTypes = 35; // 7 base types, with five channel options each (none or C1, ..., C4)

   int enum_ints[] =       {CV_8U,  CV_8UC1,  CV_8UC2,  CV_8UC3,  CV_8UC4,
                            CV_8S,  CV_8SC1,  CV_8SC2,  CV_8SC3,  CV_8SC4,
                            CV_16U, CV_16UC1, CV_16UC2, CV_16UC3, CV_16UC4,
                            CV_16S, CV_16SC1, CV_16SC2, CV_16SC3, CV_16SC4,
                            CV_32S, CV_32SC1, CV_32SC2, CV_32SC3, CV_32SC4,
                            CV_32F, CV_32FC1, CV_32FC2, CV_32FC3, CV_32FC4,
                            CV_64F, CV_64FC1, CV_64FC2, CV_64FC3, CV_64FC4};

   std::string enum_strings[] = {"CV_8U",  "CV_8UC1",  "CV_8UC2",  "CV_8UC3",  "CV_8UC4",
                                 "CV_8S",  "CV_8SC1",  "CV_8SC2",  "CV_8SC3",  "CV_8SC4",
                                 "CV_16U", "CV_16UC1", "CV_16UC2", "CV_16UC3", "CV_16UC4",
                                 "CV_16S", "CV_16SC1", "CV_16SC2", "CV_16SC3", "CV_16SC4",
                                 "CV_32S", "CV_32SC1", "CV_32SC2", "CV_32SC3", "CV_32SC4",
                                 "CV_32F", "CV_32FC1", "CV_32FC2", "CV_32FC3", "CV_32FC4",
                                 "CV_64F", "CV_64FC1", "CV_64FC2", "CV_64FC3", "CV_64FC4"};
   int typ = a.type();
   for(int i=0; i<numImgTypes; i++)
      if (typ == enum_ints[i]) return enum_strings[i];
   return "unknown image type";
}

std::string trim(std::string str, std::string chars)
//-------------------------------------
{
   if (str.length() == 0)
      return str;
   unsigned long b = str.find_first_not_of(chars);
   unsigned long e = str.find_last_not_of(chars);
   if (b == std::string::npos) return "";
   str = std::string(str, b, e - b + 1);
   return str;
}

std::size_t split(std::string s, std::vector<std::string>& tokens, std::string delim)
//-----------------------------------------------------------------------------------
{
   tokens.clear();
   std::size_t pos = s.find_first_not_of(delim);
   while (pos != std::string::npos)
   {
      std::size_t next = s.find_first_of(delim, pos);
      if (next == std::string::npos)
      {
         tokens.emplace_back(trim(s.substr(pos)));
         break;
      }
      else
      {
         tokens.emplace_back(trim(s.substr(pos, next-pos)));
         pos = s.find_first_not_of(delim, next);
      }
   }
   return tokens.size();
}


void shift(cv::Mat &img, int shift, bool is_right, cv::Mat &out)
//--------------------------------------------------------------
{
   img.copyTo(out);
   if (is_right)
      img(cv::Rect(0, 0, img.cols-shift, img.rows)).copyTo(out(cv::Rect(shift, 0, img.cols-shift, img.rows)));
   else
      img(cv::Rect(shift, 0, img.cols-shift, img.rows)).copyTo(out(cv::Rect(0, 0, img.cols-shift, img.rows)));
}