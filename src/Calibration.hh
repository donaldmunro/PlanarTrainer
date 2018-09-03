#ifndef CALIBRATION_HPP_13e0c05f_3feb_4044_8590_91faa2cfc1b0
#define CALIBRATION_HPP_13e0c05f_3feb_4044_8590_91faa2cfc1b0

#include <iostream>
#include <fstream>
#include <limits>
#ifdef FILESYSTEM_EXPERIMENTAL
#include <experimental/filesystem>
namespace filesystem = std::experimental::filesystem;
#elif defined(STD_FILESYSTEM)
#include <filesystem>
namespace filesystem = std::filesystem;
#else
#include <boost/filesystem>
#endif

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <yaml-cpp/yaml.h>

#include "util.h"
#include "ImageMeta.h"

inline bool is_opencv_yaml(filesystem::path metafile)
//-------------------------------------------------------
{
   std::string ext = metafile.extension();
   std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
   if (ext.substr(0, 5) == ".yaml")
   {
      std::ifstream ifs(metafile.string());
      std::string content((std::istreambuf_iterator<char>(ifs)), (std::istreambuf_iterator<char>()));
      return !(content.substr(0, 6) != "%YAML:");
   }
   return false;
}

class Calibration
//===============
{
public:
   Calibration() : fx(std::numeric_limits<double>::quiet_NaN()), fy(std::numeric_limits<double>::quiet_NaN()),
                   cx(std::numeric_limits<double>::quiet_NaN()), cy(std::numeric_limits<double>::quiet_NaN()) {};
   Calibration(double fx, double fy, double ox, double oy, int w, int h) : fx(fx), fy(fy), cx(ox), cy(oy), image_width(w), image_height(h) {}

   Calibration(const Calibration& other) = default;
   Calibration& operator=(const Calibration &other) = default;

   bool open(std::string file, std::stringstream* errs = nullptr)
   //------------------------------------------------------------
   {
      filesystem::path calibration_file = filesystem::absolute(file);
      if (filesystem::exists(calibration_file))
      {
         std::string ext = calibration_file.extension();
         std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
         if ( (is_opencv_yaml(calibration_file)) || (ext.substr(0, 4) == ".xml") )
         {
            cv::Mat intrinsics; //  distortion, intrinsics_inverse;
//            int image_width, image_height;
//            double rms_error =0, total_error =0;
            cv::FileStorage fs(calibration_file.string(), cv::FileStorage::READ);
            cv::FileNode fn = fs["CalibrationValues"];
            fn["intrinsics"] >> intrinsics;
            fx = intrinsics.at<double>(0, 0);
            fy = intrinsics.at<double>(1, 1);
            cx = intrinsics.at<double>(2, 0);
            cy = intrinsics.at<double>(2, 1);
//            fn["distortion"] >> distortion;
            image_width = (int) fn["width"];
            image_height = (int) fn["height"];
//            rms_error = (double) fn["rms_error"];
//            total_error = (double) fn["total_error"];
            if ( (image_width <= 0) || (image_height <= 0) )
               image_size(calibration_file);
            return  ( (fx > 0) && (fy > 0) && (cx >= 0) && (cy >= 0) );
         }
         else
         {
            if (ext.substr(0, 5) == ".yaml")
            {
               try
               {
                  YAML::Node root = YAML::LoadFile(calibration_file.c_str());
                  fx = (root["fx"]) ? root["fx"].as<double>() : std::numeric_limits<float>::quiet_NaN();
                  fy = (root["fy"]) ? root["fy"].as<double>() : std::numeric_limits<float>::quiet_NaN();
                  YAML::Node n = root["cx"];
                  if (! n)
                     n = root["ox"];
                  if (n) cx = n.as<double>();
                  n = root["cy"];
                  if (! n)
                     n = root["oy"];
                  if (n) cy = n.as<double>();
                  n = root["width"];
                  if (! n)
                     n = root["imagewidth"];
                  if (n) image_width = n.as<int>();
                  n = root["height"];
                  if (! n)
                     n = root["imageheight"];
                  if (n) image_height = n.as<int>();
               }
               catch (std::exception& e)
               {
                  if (errs != nullptr)
                     *errs << "Error reading calibration file " << filesystem::absolute(calibration_file).string();
                  return false;
               }
               if ( (image_width <= 0) || (image_height <= 0) )
                  image_size(calibration_file);
               return  ( (fx > 0) && (fy > 0) && (cx >= 0) && (cy >= 0) );
            }
         }

         // comma delimited
         std::ifstream ifs(calibration_file.string());
         std::string content((std::istreambuf_iterator<char>(ifs)), (std::istreambuf_iterator<char>()));
         content.erase(std::remove(content.begin(), content.end(), ' '), content.end());
         content.erase(std::remove(content.begin(), content.end(), '\t'), content.end());
         content.erase(std::remove(content.begin(), content.end(), '\n'), content.end());
         return set(content);

         return false;
      }
      else
         return false;
   }

   bool save(std::string file, bool is_overwrite = true, std::stringstream* errs = nullptr)
   //------------------------------------------------------------
   {
      filesystem::path calibration_file = filesystem::absolute(file);
      if ( (filesystem::exists(calibration_file)) && (! is_overwrite) )
      {
         if (errs)
            *errs << calibration_file.string() << " exists";
         return false;
      }

      std::string ext = calibration_file.extension();
      std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

      if (ext.substr(0, 4) == ".xml")
      {
         cv::FileStorage fs(calibration_file.c_str(), cv::FileStorage::WRITE);
         fs << "CalibrationValues" << "{";
         fs << "intrinsics" << asMat3x3f() << "distortion" << 0 << "width" << image_width << "height"
            << image_height << "rms_error" << 0 << "total_error" << 0 << "}";
         return true;
      }
      else
      {
         YAML::Emitter out;
         out.SetIndent(3);
         out << YAML::BeginMap;
         out << YAML::Key << "fx" << YAML::Value << fx;
         out << YAML::Key << "fy" << YAML::Value << fy;
         out << YAML::Key << "cx" << YAML::Value << cx;
         out << YAML::Key << "cy" << YAML::Value << cy;
         if ( (image_width > 0) && (image_height > 0) )
         {
            out << YAML::Key << "imagewidth" << YAML::Value << image_width;
            out << YAML::Key << "imageheight" << YAML::Value << image_height;
         }
         out << YAML::EndMap;
         std::ofstream of(calibration_file.c_str());
         of << out.c_str();
         return of.good();
      }
   }

   bool set(std::string delimited, std::stringstream* errs = nullptr)
   //--------------------------------------------------------------
   {
      std::vector<std::string> tokens;
      size_t n = split(delimited, tokens, ",");
      if (n >= 4)
      {
         try
         {
            fx = std::stod(trim(tokens[0]));
            fy = std::stod(trim(tokens[1]));
            cx = std::stod(trim(tokens[2]));
            cy = std::stod(trim(tokens[3]));
            image_width = image_height = -1;
            if (n > 4)
               image_width = std::stoi(tokens[4]);
            if (n > 5)
               image_height = std::stoi(tokens[5]);

         }
         catch (std::exception &e)
         {
            if (errs != nullptr)
               *errs << "Invalid (non numeric) values in comma delimited calibration values " << " (" << tokens[0] << ","
                     << tokens[1] << "," << tokens[2] << "," << tokens[3] << ")";
            return false;
         }
         return true;
      }
      return false;
   }

   cv::Matx33f asMatx3x3f() { return cv::Matx33f(fx, 0, cx, 0, fy, cy, 0, 0, 1); }

   cv::Matx33f asMatx3x3d() { return cv::Matx33d(fx, 0, cx, 0, fy, cy, 0, 0, 1); }

   cv::Mat asMat3x3d()
   {
      cv::Mat K = cv::Mat::eye(3, 3, CV_64FC1);
      K.at<double>(0, 0) = fx;
      K.at<double>(1, 1) = fy;
      K.at<double>(0, 2) = cx;
      K.at<double>(1, 2) = cy;
      return K;
   }

   cv::Mat asMat3x3f()
   {
      cv::Mat K = cv::Mat::eye(3, 3, CV_32FC1);
      K.at<float>(0, 0) = static_cast<float>(fx);
      K.at<float>(1, 1) = static_cast<float>(fy);
      K.at<float>(0, 2) = static_cast<float>(cx);
      K.at<float>(1, 2) = static_cast<float>(cy);
      return K;
   }

private:
   double fx, fy, cx, cy;
   int image_width =-1, image_height =-1;

   void image_size(filesystem::path& calibration_path)
   //------------------------------------------------
   {
      filesystem::path base = calibration_path.parent_path() / calibration_path.stem();
      filesystem::path imgfile(base.string() + ".png");
      cv::Mat img;
      if (filesystem::exists(imgfile))
         img = cv::imread(imgfile.c_str());
      if (img.empty())
      {
         imgfile = filesystem::path(base.string() + ".jpg");
         if (filesystem::exists(imgfile))
            img = cv::imread(imgfile.c_str());
      }
      if (img.empty())
      {
         imgfile = filesystem::path(base.string() + ".jpeg");
         if (filesystem::exists(imgfile))
            img = cv::imread(imgfile.c_str());
      }
      if (! img.empty())
      {
         image_width = img.cols;
         image_height = img.rows;
      }
   }
};
#endif //TRAINER_CALIBRATION_HPP
