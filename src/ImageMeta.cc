#include <iostream>
#include <fstream>
#include <math.h>

#include <opencv2/core/core.hpp>

#include "ImageMeta.h"

namespace imeta
{
   ImageMeta::ImageMeta(std::string filename) { open(filename); }

   bool ImageMeta::open(const filesystem::path &f)
   //----------------------------------------------------
   {
      is_good = false;
      device_rotation =-1;
      latitude =std::numeric_limits<float>::quiet_NaN();
      longitude =std::numeric_limits<float>::quiet_NaN();
      altitude =std::numeric_limits<float>::quiet_NaN(), z_deviation =0, depth =0;
      has_location = false;
      deviceGravity.release(), correctedGravity.release(), rotation_vector.release();
      cv::FileStorage fs;
      metafile = get_metafile(f);
      if (metafile.empty()) return false;
      try
      {
         bool is_open;
         if (! ImageMeta::is_opencv_yaml(metafile))
            return open_yaml(metafile);
         try { is_open = fs.open(metafile.c_str(), cv::FileStorage::READ); } catch (cv::Exception& cverr) { is_open = false; }
         if (! is_open)
         {
            log << "Error opening image meta file " << metafile.c_str() << std::endl;
            is_good = false;
            return false;
         }
         cv::FileNode node = fs["deviceRotation"];
         if ((!node.empty()) && (node.isInt()))
            device_rotation = (int) node;
         latitude = longitude = std::numeric_limits<float>::quiet_NaN();
         try
         {
            node = fs["latitude"];
            if ((!node.empty()) && (node.isReal()))
               latitude = (float) node;
            node = fs["longitude"];
            if ((!node.empty()) && (node.isReal()))
               longitude = (float) node;
            has_location = ( (! isnan(latitude)) && (! isnan(longitude)) );
         }
         catch (std::exception &e)
         {
            latitude = longitude = std::numeric_limits<float>::quiet_NaN();
            has_location = false;
         }
         try
         {
            node = fs["altitude"];
            if ( (!node.empty()) && (node.isReal()) )
               altitude = (float) node;
         }
         catch (std::exception &e)
         {
            altitude = 0;
         }
         try
         {
            node = fs["depth"];
            if ( (!node.empty()) && (node.isReal()) )
               depth = (float) node;
         }
         catch (std::exception &e)
         {
            depth = 0;
         }
         deviceGravity.release();
         correctedGravity.release();
         try
         {
            node = fs["rawGravity"];
            if (! node.empty())
               node >> deviceGravity;
         }
         catch (std::exception &e)
         {}
         try
         {
            node = fs["gravity"];
            if (! node.empty())
               node >> correctedGravity;
         }
         catch (std::exception &e)
         {}
         rotation_vector.release();
         try
         {
            node = fs["rotationVec"];
            if (! node.empty())
               node >> rotation_vector;
            std::stringstream ss;
            ss << rotation_vector;
            node = fs["z_deviation"];
            if (! node.empty())
               z_deviation = (float) node;
         }
         catch (std::exception &e)
         {
         }
         is_good = ((!deviceGravity.empty()) || (!correctedGravity.empty()));
      }
      catch (std::exception &e)
      {
         log << "Error deserializing " << metafile.c_str() << " (" << e.what() << ")";
         std::cerr << log.str() << std::endl;
         is_good = false;
         return false;
      }
      return is_good;
   }

   bool ImageMeta::open_yaml(filesystem::path metafile)
   //--------------------------------------------------
   {
      try
      {
         calibration.reset(nullptr);
         YAML::Node root = YAML::LoadFile(metafile.c_str());
         device_rotation = root["deviceRotation"].as<int>();
         YAML::Node n = root["latitude"];
         latitude = (n) ? n.as<float>() : std::numeric_limits<float>::quiet_NaN();
         n = root["longitude"];
         longitude = (n) ? n.as<float>() : std::numeric_limits<float>::quiet_NaN();
         has_location = ( (! isnan(latitude)) && (! isnan(longitude)) );
         n = root["altitude"];
         altitude = (n) ? n.as<float>() : std::numeric_limits<float>::quiet_NaN();
         n = root["depth"];
         depth = (n) ? n.as<float>() : 0;
         n = root["rawGravity"];
         if (n)
            imeta::yaml_mat<double>(1, 3, n, deviceGravity);
         else
            deviceGravity.release();
         n = root["gravity"];
         if (n)
            imeta::yaml_mat<double>(1, 3, n, correctedGravity);
         else
            correctedGravity.release();
         n = root["rotationVec"];
         if (n)
            imeta::yaml_mat<double>(1, 5, n, rotation_vector);
         else
            rotation_vector.release();
         n = root["z_deviation"];
         if (n)
            z_deviation = n.as<float>();
         double fx = (root["fx"]) ? root["fx"].as<double>() : -1;
         double fy = (root["fy"]) ? root["fy"].as<double>() : -1;
         double cx =-1, cy = -1;
         n = root["cx"];
         if (! n) n = root["ox"];
         if (n) cx = n.as<double>();
         n = root["cy"];
         if (! n) n = root["oy"];
         if (n) cy = n.as<double>();
         if ( (fx > 0) && (fy > 0) && (cx >= 0) && (cy >= 0) )
         {
            int w =-1, h =-1;
            n = root["imagewidth"];
            if (! n) n = root["width"];
            if (n) w = n.as<int>();
            n = root["imageheight"];
            if (! n) n = root["height"];
            if (n) h = n.as<int>();
            calibration.reset(new Calibration(fx, fy, cx, cy, w, h));
         }
      }
      catch (std::exception &e)
      {
         log << "Error deserializing " << metafile.c_str() << " (" << e.what() << ")";
         std::cerr << log.str() << std::endl;
         is_good = false;
         return false;
      }
      is_good = ((!deviceGravity.empty()) || (!correctedGravity.empty()));
      return is_good;
   }

   filesystem::path ImageMeta::get_metafile(filesystem::path metafile)
   //-----------------------------------------------------------------
   {
      std::string ext = metafile.extension();
      std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
      if ((filesystem::is_regular_file(metafile)) && ((ext == ".yaml") || (ext == ".xml")))
         return filesystem::absolute(metafile);
      metafile = metafile.parent_path() / filesystem::path(metafile.stem().string() + ".yaml");
      if (filesystem::is_regular_file(metafile))
         return filesystem::absolute(metafile);
      metafile = metafile.parent_path() / filesystem::path(metafile.stem().string() + ".xml");
      if (filesystem::is_regular_file(metafile))
         return filesystem::absolute(metafile);
      return filesystem::path();
   }

   bool ImageMeta::is_opencv_yaml(filesystem::path metafile)
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


}
