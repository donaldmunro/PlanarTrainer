#include <sys/stat.h>

#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <functional>

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

#define RAPIDJSON_HAS_STDSTRING 1
#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>

#include "yaml-cpp/yaml.h"

#include "ImageMeta.h"
#include "util.h"
#include "json.h"


bool json_write(const cv::Mat& img, const cv::Mat& imgRGB, const std::vector<cv::KeyPoint>& k, const cv::Mat& d,
                const std::vector<cv::KeyPoint>& hull_keypoints, const cv::Mat& hull_descriptors, const cv::Point2f& centroid,
//              const std::vector<barycentric_info>& barycentrics, int max_triangles, float min_area_perc,
                const std::string& detector_name, const std::string& descriptor_name,
                int id, int rid, const std::string& name, std::string dir,
                double latitude, double longitude, float altitude, double depth, const cv::Rect2f& bb, const cv::RotatedRect& rbb,
                const imeta::ImageMeta& image_meta)
//------------------------------------------------------------------------------------------------------------------------
{
   if (k.size() <= 0)
   {
      std::cerr << "Error computing descriptors" << std::endl;
      return false;
   }
   dir = trim(dir);
   if (dir.empty())
   {
      std::stringstream ss;
      ss << id << "-" << name;
      dir = ss.str();
      filesystem::path directory(dir);
      if (! filesystem::is_directory(directory))
         ::mkdir(dir.c_str(), 0660);
   }
   else
   {
      filesystem::path directory(dir);
      if (! filesystem::is_directory(directory))
         ::mkdir(dir.c_str(), 0660);
   }
   std::stringstream ss;
   ss << dir << "/" << rid;
   std::string jsonpath = ss.str() + ".json";
   std::string imgpath = ss.str() + ".png";
   std::string rgbpath = ss.str() + "-color.png";
   cv::imwrite(imgpath, img);
   cv::imwrite(rgbpath, imgRGB);

   rapidjson::Document document;
   rapidjson::Document::AllocatorType& allocator = document.GetAllocator();
   document.SetObject();

   const int typ = d.type();
   const std::string type_name = type(d);
   document.AddMember("id", id, allocator);
   document.AddMember("rid", rid, allocator);
   document.AddMember("name", rapidjson::Value().SetString(name, allocator), allocator);
   document.AddMember("detector", rapidjson::Value().SetString(detector_name, allocator), allocator);
   document.AddMember("descriptor", rapidjson::Value().SetString(descriptor_name, allocator), allocator);
   document.AddMember("MatType", typ, allocator);
   document.AddMember("MatTypeName", rapidjson::Value().SetString(type_name, allocator), allocator);
   document.AddMember("ImgMono", rapidjson::Value().SetString(imgpath, allocator), allocator);
   document.AddMember("ImgColor", rapidjson::Value().SetString(rgbpath, allocator), allocator);
   document.AddMember("rows", img.rows, allocator);
   document.AddMember("cols", img.cols, allocator);
   const std::function<rapidjson::Value(const cv::KeyPoint&, rapidjson::Document::AllocatorType& allocator)> kpfn
                                                                                         = &jsoncv::encode_ocv_keypoint;
   document.AddMember("keypoints", jsoncv::encode_vector(k, allocator, kpfn), allocator);
   document.AddMember("descriptors", jsoncv::encode_ocv_mat(d, allocator), allocator);
   document.AddMember("hkeypoints", jsoncv::encode_vector(hull_keypoints, allocator,  kpfn), allocator);
   document.AddMember("hdescriptors", jsoncv::encode_ocv_mat(hull_descriptors, allocator), allocator);
   document.AddMember("centreX", centroid.x, allocator);
   document.AddMember("centreY", centroid.y, allocator);
   if (image_meta)
   {
      if ( (image_meta.has_location) && (! std::isnan(image_meta.latitude)) && (! std::isnan(image_meta.longitude)) )
      {
         latitude = image_meta.latitude;
         longitude = image_meta.longitude;
      }
      if ( (! std::isnan(image_meta.altitude)) && (image_meta.altitude > 0) && (! std::isnan(image_meta.altitude)) )
         altitude = image_meta.altitude;

      document.AddMember("z_deviation", image_meta.z_deviation, allocator);
      if (! image_meta.deviceGravity.empty())
         document.AddMember("device_gravity", jsoncv::encode_ocv_mat(image_meta.deviceGravity, allocator), allocator);
      if (! image_meta.correctedGravity.empty())
         document.AddMember("gravity", jsoncv::encode_ocv_mat(image_meta.correctedGravity, allocator), allocator);
      if (! image_meta.rotation_vector.empty())
         document.AddMember("rotationVec", jsoncv::encode_ocv_mat(image_meta.rotation_vector, allocator), allocator);
      if ( (! std::isnan(image_meta.depth)) && (image_meta.depth > 0) && (! std::isnan(image_meta.depth)) )
         depth = image_meta.depth;
   }
   document.AddMember("latitude", latitude, allocator);
   document.AddMember("longitude", longitude, allocator);
   document.AddMember("altitude", altitude, allocator);
   if (bb.area() > 0)
   {
      rapidjson::Value v(rapidjson::kObjectType);
      v.AddMember("x", rapidjson::Value().SetFloat(bb.x), allocator);
      v.AddMember("y", rapidjson::Value().SetFloat(bb.y), allocator);
      v.AddMember("width", rapidjson::Value().SetFloat(bb.width), allocator);
      v.AddMember("height", rapidjson::Value().SetFloat(bb.height), allocator);
      document.AddMember("bb", v, allocator);
   }
   if (rbb.size.area() > 0)
   {
      rapidjson::Value v(rapidjson::kObjectType);
      v.AddMember("cx", rapidjson::Value().SetFloat(rbb.center.x), allocator);
      v.AddMember("cy", rapidjson::Value().SetFloat(rbb.center.y), allocator);
      v.AddMember("width", rapidjson::Value().SetFloat(rbb.size.width), allocator);
      v.AddMember("height", rapidjson::Value().SetFloat(rbb.size.height), allocator);
      v.AddMember("angle", rapidjson::Value().SetFloat(rbb.angle), allocator);
      document.AddMember("rotated_bb", v, allocator);
   }
   if ( (depth > 0) && (! std::isnan(depth)) )
      document.AddMember("depth", depth, allocator);
   else
      document.AddMember("depth", 0, allocator);
   std::ofstream of(jsonpath);
   rapidjson::StringBuffer strbuf;
   rapidjson::Writer<rapidjson::StringBuffer> writer(strbuf);
//   rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(strbuf);
//   writer.SetIndent(' ', 2);
   document.Accept(writer);
   of << strbuf.GetString() << std::endl;
   return of.good();
}

inline void ocv_write_keypoints(cv::FileStorage& fs, const std::vector<cv::KeyPoint>& keypoints)
//----------------------------------------------------------------------------------------------
{
   std::stringstream ss;
   fs << "[";
   for (int i=0; i<(int)keypoints.size(); i++)
   {
      const cv::KeyPoint &keypoint = keypoints.at(i);
      ss.str("");
      ss << std::fixed << std::setprecision(8) << keypoint.pt.x << "," << keypoint.pt.y << "," << keypoint.size
         << "," << keypoint.angle << "," << keypoint.response << "," << keypoint.octave << "," << keypoint.class_id;
      fs << ss.str();
   }
   fs << "]";
}

inline void ocv_write_matches(cv::FileStorage& fs, const std::vector<cv::DMatch>& matches)
//----------------------------------------------------------------------------------------------
{
   std::stringstream ss;
   fs << "[";
   for (int i=0; i<(int)matches.size(); i++)
   {
      const cv::DMatch &match = matches.at(i);
      ss.str("");
      ss << std::fixed << std::setprecision(8) << match.trainIdx << "," << match.queryIdx << "," << match.imgIdx
         << "," << match.distance;
      fs << ss.str();
   }
   fs << "]";
}


bool ocv_write(const cv::Mat& img, const cv::Mat& imgRGB, const std::vector<cv::KeyPoint>& k, const cv::Mat& d,
               const std::vector<cv::KeyPoint>& hull_keypoints, const cv::Mat& hull_descriptors, const cv::Point2f& centroid,
//              const std::vector<barycentric_info>& barycentrics, int max_triangles, float min_area_perc,
               const std::string& detector_name, const std::string& descriptor_name,
               int id, int rid, const std::string& name, std::string dir,
               double latitude, double longitude, float altitude, double depth, const cv::Rect2f& bb, const cv::RotatedRect& rbb,
               const imeta::ImageMeta& image_meta)
//--------------------------------------------------------------------------------------------------
{
   if (k.size() <= 0)
   {
      std::cerr << "Error computing descriptors" << std::endl;
      return false;
   }
   dir = trim(dir);
   if (dir.empty())
   {
      std::stringstream ss;
      ss << id << "-" << name;
      dir = ss.str();
      filesystem::path directory(dir);
      if (! filesystem::is_directory(directory))
         ::mkdir(dir.c_str(), 0660);
   }
   else
   {
      filesystem::path directory(dir);
      if (! filesystem::is_directory(directory))
         ::mkdir(dir.c_str(), 0660);
   }
   std::stringstream ss;
   ss << dir << "/" << rid;
   std::string xmlpath = ss.str() + ".xml";
   std::string imgpath = ss.str() + ".png";
   std::string rgbpath = ss.str() + "-color.png";
   cv::imwrite(imgpath, img);
   cv::imwrite(rgbpath, imgRGB);

   const int typ = d.type();
   const std::string type_name = type(d);
   cv::FileStorage fs(xmlpath, cv::FileStorage::WRITE);
   fs << "id" << id << "rid" << rid << "name" << name << "detector" << detector_name << "descriptor"
      << descriptor_name << "MatType" << typ << "MatTypeName" << type_name << "ImgMono"  << imgpath <<
      "ImgColor" << rgbpath << "rows" << img.rows << "cols" << img.cols << "keypoints";
   ocv_write_keypoints(fs, k);
   fs << "descriptors" << d;
   fs << "hkeypoints";
   ocv_write_keypoints(fs, hull_keypoints);
   fs << "hdescriptors" << hull_descriptors;

   fs << "centre" << centroid;
   if (image_meta)
   {
      fs << "device_rotation" << image_meta.device_rotation;
      if (image_meta.has_location)
      {
         fs << "latitude" << image_meta.latitude;
         fs << "longitude" << image_meta.longitude;
      }
      else
      {
         if ( (! std::isnan(latitude)) && (! std::isnan(longitude)) )
         {
            fs << "latitude" << latitude;
            fs << "longitude" << longitude;
         }
      }
      if (! std::isnan(image_meta.altitude))
         fs << "altitude" << image_meta.altitude;
      else if (! std::isnan(altitude))
         fs << "altitude" << altitude;
      if (! image_meta.deviceGravity.empty())
         fs << "device_gravity" << image_meta.deviceGravity;
      if (! image_meta.correctedGravity.empty())
         fs << "correctedGravity" << image_meta.correctedGravity;
      if (! image_meta.rotation_vector.empty())
         fs << "rotationVec" << image_meta.rotation_vector;
      fs << "z_deviation" << image_meta.z_deviation;
      if ( (! std::isnan(image_meta.depth)) && (image_meta.depth > 0) && (! std::isnan(image_meta.depth)) )
         depth = image_meta.depth;
   }
   else
   {
      if ( (! std::isnan(latitude)) && (! std::isnan(longitude)) )
      {
         fs << "latitude" << latitude;
         fs << "longitude" << longitude;
      }
      if (! std::isnan(altitude))
         fs << "altitude" << altitude;
   }
   if (bb.area() > 0)
   {
      std::vector<float> v = { bb.x, bb.y, bb.width, bb.height };
      fs << "bb" << v;
   }
   if (rbb.size.area() > 0)
   {
      std::vector<float> v = { rbb.center.x, rbb.center.y, rbb.angle, rbb.size.width, rbb.size.height };
      fs << "rotated_bb" << v;
   }
   if ( (depth > 0) && (! std::isnan(depth)) )
      fs << "depth" << depth;
   else
      fs << "depth" << 0.0;
   std::cout << "Wrote " << xmlpath << std::endl;
   return true;
}

template<typename T>
inline YAML::Emitter& yaml_map(YAML::Emitter& out, std::string k, T v)
//--------------------------------------------------------------------
{
   out << YAML::Key << k << YAML::Value << v;
   return out;
}

YAML::Emitter& operator<< (YAML::Emitter& out, const cv::KeyPoint& kp)
//--------------------------------------------------------------------
{
   out << YAML::BeginMap;
   out << YAML::Key << "x" << YAML::Value << kp.pt.x;
   out << YAML::Key << "y" << YAML::Value << kp.pt.y;
   out << YAML::Key << "response" << YAML::Value << kp.response;
   out << YAML::Key << "angle" << YAML::Value << kp.angle;
   out << YAML::Key << "size" << YAML::Value << kp.size;
   out << YAML::Key << "octave" << YAML::Value << kp.octave;
   out << YAML::EndMap;
   return out;
}

YAML::Emitter& operator<< (YAML::Emitter& out, const cv::Point3d& pt)
//--------------------------------------------------------------------
{
   out << YAML::BeginMap;
   out << YAML::Key << "x" << YAML::Value << pt.x;
   out << YAML::Key << "y" << YAML::Value << pt.y;
   out << YAML::Key << "z" << YAML::Value << pt.z;
   out << YAML::EndMap;
   return out;
}

YAML::Emitter& operator<< (YAML::Emitter& out, const std::vector<cv::KeyPoint>& kps)
//----------------------------------------------------------------------------------
{
   out << YAML::BeginSeq;
   for (const cv::KeyPoint& kp : kps)
      out << kp;
   out << YAML::EndSeq;
   return out;
}

YAML::Emitter& operator<< (YAML::Emitter& out, const std::vector<std::pair<cv::Point3d, cv::Point3d>>& mpts)
{
   out << YAML::BeginSeq;
   for (const std::pair<cv::Point3d, cv::Point3d> pp : mpts)
   {
      out << YAML::BeginMap;
      out << YAML::Key << "first" << YAML::Value << pp.first;
      out << YAML::Key << "second" << YAML::Value << pp.second;
      out << YAML::EndMap;
   }
   out << YAML::EndSeq;
   return out;
}

YAML::Emitter& operator<< (YAML::Emitter& out, const cv::Rect2f& rect)
//------------------------------------------------------------------
{
   out << YAML::BeginMap;
   out << YAML::Key << "x" << YAML::Value << rect.x << YAML::Key << "y" << YAML::Value << rect.y
       << YAML::Key << "width" << YAML::Value << rect.width << YAML::Key << "height" << YAML::Value << rect.height;
   out << YAML::EndMap;
   return out;
}

YAML::Emitter& operator<< (YAML::Emitter& out, const cv::RotatedRect& rbb)
//------------------------------------------------------------------
{
   out << YAML::BeginMap;
   out << YAML::Key << "cx" << YAML::Value << rbb.center.x << YAML::Key << "cy" << YAML::Value << rbb.center.y
         << YAML::Key << "width" << YAML::Value << rbb.size.width << YAML::Key << "height" << YAML::Value << rbb.size.height
         << YAML::Key << "angle" << YAML::Value << rbb.angle;
   out << YAML::EndMap;
   return out;
}

YAML::Emitter& operator<< (YAML::Emitter& out, const cv::DMatch& match)
//------------------------------------------------------------------
{
   out << YAML::BeginMap;
   out << YAML::Key << "trainIdx" << YAML::Value << match.trainIdx << YAML::Key << "queryIdx" << YAML::Value << match.queryIdx
       << YAML::Key << "distance" << YAML::Value << match.distance << YAML::Key << "imgIdx" << YAML::Value << match.imgIdx;
   out << YAML::EndMap;
   return out;
}

YAML::Emitter& operator<< (YAML::Emitter& out, const std::vector<cv::DMatch>& matches)
//----------------------------------------------------------------------------------
{
   out << YAML::BeginSeq;
   for (const cv::DMatch& m : matches)
      out << m;
   out << YAML::EndSeq;
   return out;
}

YAML::Emitter& operator<< (YAML::Emitter& out, const cv::Mat& m)
//--------------------------------------------------------------
{
   std::string typname = type(m);
   switch (m.type())
   {
      case CV_32FC1:
      case CV_64FC1:
      case CV_8U:
         break;
      default:
         std::stringstream ss;
         ss << "YAML::Emitter& operator<< (YAML::Emitter& out, const cv::Mat& m) does not support "
            << typname << "  yet";
         std::cerr << ss.str() << std::endl;
         throw std::logic_error(ss.str());
   }
   out << YAML::BeginMap;
   yaml_map(out, "type",  m.type());
   yaml_map(out, "typename", typname);
   yaml_map(out, "rows", m.rows);
   yaml_map(out, "cols", m.cols);
   out << YAML::Key << "data" << YAML::Value;
   out << YAML::BeginSeq;

   switch (m.type())
   {
      case CV_32FC1:
         for (int row = 0; row < m.rows; row++)
         {
            const float *ptr = m.ptr<float>(row);
            for (int col = 0; col < m.cols; col++)
               out << ptr[col];
         }
         break;
      case CV_64FC1:
         for (int row = 0; row < m.rows; row++)
         {
            const double *ptr = m.ptr<double>(row);
            for (int col = 0; col < m.cols; col++)
               out << ptr[col];
         }
         break;
      case CV_8U:
         for (int row = 0; row < m.rows; row++)
         {
            const uchar *ptr = m.ptr<uchar>(row);
            for (int col = 0; col < m.cols; col++)
               out << static_cast<unsigned>(ptr[col]);
         }
         break;
      default:
         throw std::logic_error(typname + " not supported");
   }
   out << YAML::EndSeq;
   out << YAML::EndMap;
   return out;
}


bool yaml_write(const cv::Mat& img, const cv::Mat& imgRGB, const std::vector<cv::KeyPoint>& k, const cv::Mat& d,
                const std::vector<cv::KeyPoint>& hull_keypoints, const cv::Mat& hull_descriptors, const cv::Point2f& centroid,
//              const std::vector<barycentric_info>& barycentrics, int max_triangles, float min_area_perc,
                const std::string& detector_name, const std::string& descriptor_name,
                int id, int rid, const std::string& name, std::string dir,
                double latitude, double longitude, float altitude, double depth, const cv::Rect2f& bb, const cv::RotatedRect& rbb,
                const imeta::ImageMeta& image_meta)
//---------------------------------------------------------------------------------------------------------------------------------
{
   if (k.size() <= 0)
   {
      std::cerr << "Error computing descriptors" << std::endl;
      return false;
   }
   dir = trim(dir);
   if (dir.empty())
   {
      std::stringstream ss;
      ss << id << "-" << name;
      dir = ss.str();
      filesystem::path directory(dir);
      if (! filesystem::is_directory(directory))
         ::mkdir(dir.c_str(), 0660);
   }
   else
   {
      filesystem::path directory(dir);
      if (! filesystem::is_directory(directory))
         ::mkdir(dir.c_str(), 0660);
   }
   std::stringstream ss;
   ss << dir << "/" << rid;
   std::string yamlpath = ss.str() + ".yaml";
   std::string imgpath = ss.str() + ".png";
   std::string rgbpath = ss.str() + "-color.png";
   cv::imwrite(imgpath, img);
   cv::imwrite(rgbpath, imgRGB);

   YAML::Emitter out;
   out.SetIndent(3);
   const int typ = d.type();
   const std::string type_name = type(d);
   out << YAML::BeginMap;
   yaml_map(out, "id", id);
   yaml_map(out, "rid", rid);
   yaml_map(out, "name", name);
   yaml_map(out, "detector",detector_name);
   yaml_map(out, "descriptor",descriptor_name);
   yaml_map(out, "MatType",typ);
   yaml_map(out, "MatTypeName",type_name);
   yaml_map(out, "ImgMono",imgpath);
   yaml_map(out, "ImgColor",rgbpath);
   yaml_map(out, "rows", img.rows);
   yaml_map(out, "cols", img.cols);
   yaml_map(out, "centreX", centroid.x);
   yaml_map(out, "centreY", centroid.y);
   if (image_meta)
   {
      if ( (image_meta.has_location) && (! std::isnan(image_meta.latitude)) && (! std::isnan(image_meta.longitude)) )
      {
         latitude = image_meta.latitude;
         longitude = image_meta.longitude;
      }
      if ( (! std::isnan(image_meta.altitude)) && (image_meta.altitude > 0) && (! std::isnan(image_meta.altitude)) )
         altitude = image_meta.altitude;
      if ( (! std::isnan(image_meta.depth)) && (image_meta.depth > 0) && (! std::isnan(image_meta.depth)) )
         depth = image_meta.depth;

      yaml_map(out, "z_deviation", image_meta.z_deviation);
      if (! image_meta.deviceGravity.empty())
         yaml_map(out, "device_gravity", image_meta.deviceGravity);
      if (! image_meta.correctedGravity.empty())
         yaml_map(out, "gravity", image_meta.correctedGravity);
      if (! image_meta.rotation_vector.empty())
         yaml_map(out, "rotationVec", image_meta.rotation_vector);
   }
   yaml_map(out, "latitude", latitude);
   yaml_map(out, "longitude", longitude);
   yaml_map(out, "altitude", altitude);
   if ( (depth > 0) && (! std::isnan(depth)) )
      yaml_map(out, "depth", depth);
   else
      yaml_map(out, "depth", 0.0);
   if (bb.area() > 0)
      yaml_map(out, "bb", bb);
   if (rbb.size.area() > 0)
      yaml_map(out, "rotated_bb", rbb);

   out << YAML::Key << "keypoints" << YAML::Value << k;
   out << YAML::Key << "descriptors" << YAML::Value << d;
   out << YAML::Key << "hkeypoints" << YAML::Value << hull_keypoints;
   out << YAML::Key << "hdescriptors" << YAML::Value << hull_descriptors;

   out << YAML::EndMap;
   std::ofstream of(yamlpath);
   of << out.c_str();
   return of.good();
}

bool ocv_write_matches(std::string dir, const std::string& name, const std::vector<cv::KeyPoint>& matched_train_keypoints,
                       const cv::Mat& matched_train_descriptors,
                       const std::vector<cv::KeyPoint>& matched_query_keypoints, const cv::Mat& matched_query_descriptors,
                       const std::vector<cv::KeyPoint>& homography_train_keypoints, const cv::Mat& homography_train_descriptors,
                       const std::vector<cv::KeyPoint>& homography_query_keypoints, const cv::Mat& homography_query_descriptors,
                       const std::vector<cv::DMatch>& matches, const std::vector<cv::DMatch>& homography_matches,
                       std::stringstream *errs)
//------------------------------------------------------------------------------------------------------------------------
{
   dir = trim(dir);
   if (dir.empty())
      return false;
   else
   {
      filesystem::path directory(dir);
      if ( (filesystem::exists(directory)) && (! filesystem::is_directory(directory)) )
      {
         if (errs != nullptr)
            *errs << directory.string() << "exists but is not a directory";
         std::cerr << "ocv_write_matches: " << directory.string() << " exists but is not a directory";
         return false;
      }
      if (! filesystem::exists(directory))
         filesystem::create_directories(directory);
   }
   std::stringstream ss;
   ss << dir << "/" << name << "-matches.xml";
   cv::FileStorage fs(ss.str(), cv::FileStorage::WRITE);
   fs << "matched_train_keypoints"; ocv_write_keypoints(fs, matched_train_keypoints);
   fs << "matched_train_descriptors" << matched_train_descriptors;
   fs << "matched_query_keypoints"; ocv_write_keypoints(fs, matched_query_keypoints);
   fs << "matched_query_descriptors" << matched_query_descriptors;
   fs << "homography_train_keypoints"; ocv_write_keypoints(fs, homography_train_keypoints);
   fs << "homography_train_descriptors" << homography_train_descriptors;
   fs << "homography_query_keypoints"; ocv_write_keypoints(fs, homography_query_keypoints);
   fs << "homography_query_descriptors" << homography_query_descriptors;
   fs << "matches"; ocv_write_matches(fs, matches);
   fs << "homography_matches"; ocv_write_matches(fs, homography_matches);
   std::cout << "Wrote matches.xml to" << dir << std::endl;
   return true;
}

bool json_write_matches(std::string dir, const std::string& name,
                        const std::vector<cv::KeyPoint>& matched_train_keypoints, const cv::Mat& matched_train_descriptors,
                        const std::vector<cv::KeyPoint>& matched_query_keypoints, const cv::Mat& matched_query_descriptors,
                        const std::vector<cv::KeyPoint>& homography_train_keypoints, const cv::Mat& homography_train_descriptors,
                        const std::vector<cv::KeyPoint>& homography_query_keypoints, const cv::Mat& homography_query_descriptors,
                        const std::vector<cv::DMatch>& matches, const std::vector<cv::DMatch>& homography_matches,
                        std::stringstream *errs)
//---------------------------------------------------------------------------------------------------------------------------------------
{
   dir = trim(dir);
   if (dir.empty())
      return false;
   else
   {
      filesystem::path directory(dir);
      if ((filesystem::exists(directory)) && (!filesystem::is_directory(directory)))
      {
         if (errs != nullptr)
            *errs << directory.string() << "exists but is not a directory";
         std::cerr << "yaml_write_matches: " << directory.string() << " exists but is not a directory";
         return false;
      }
      if (!filesystem::exists(directory))
         filesystem::create_directories(directory);
   }
   std::stringstream ss;
   ss << dir << "/" << name << "-matches.json";

   const std::function<rapidjson::Value(const cv::KeyPoint&, rapidjson::Document::AllocatorType& allocator)> kpfn
         = &jsoncv::encode_ocv_keypoint;
   const std::function<rapidjson::Value(const cv::DMatch&, rapidjson::Document::AllocatorType& allocator)> mfn
         = &jsoncv::encode_ocv_match;
   rapidjson::Document document;
   rapidjson::Document::AllocatorType& allocator = document.GetAllocator();
   document.SetObject();
   document.AddMember("matched_train_keypoints", jsoncv::encode_vector(matched_train_keypoints, allocator, kpfn), allocator);
   document.AddMember("matched_train_descriptors", jsoncv::encode_ocv_mat(matched_train_descriptors, allocator), allocator);
   document.AddMember("matched_query_keypoints", jsoncv::encode_vector(matched_query_keypoints, allocator, kpfn), allocator);
   document.AddMember("matched_query_descriptors", jsoncv::encode_ocv_mat(matched_query_descriptors, allocator), allocator);
   document.AddMember("homography_train_keypoints", jsoncv::encode_vector(homography_train_keypoints, allocator, kpfn), allocator);
   document.AddMember("homography_train_descriptors", jsoncv::encode_ocv_mat(homography_train_descriptors, allocator), allocator);
   document.AddMember("homography_query_keypoints", jsoncv::encode_vector(homography_query_keypoints, allocator, kpfn), allocator);
   document.AddMember("homography_query_descriptors", jsoncv::encode_ocv_mat(homography_query_descriptors, allocator), allocator);
   document.AddMember("matches", jsoncv::encode_vector(matches, allocator, mfn), allocator);
   document.AddMember("homography_matches", jsoncv::encode_vector(homography_matches, allocator, mfn), allocator);
   std::ofstream of(ss.str());
   rapidjson::StringBuffer strbuf;
   rapidjson::Writer<rapidjson::StringBuffer> writer(strbuf);
//   rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(strbuf);
//   writer.SetIndent(' ', 2);
   document.Accept(writer);
   of << strbuf.GetString() << std::endl;
   std::cout << "Wrote matches.xml to" << dir << std::endl;
   return true;
}

bool yaml_write_matches(std::string dir, const std::string& name,
                        const std::vector<cv::KeyPoint>& matched_train_keypoints, const cv::Mat& matched_train_descriptors,
                        const std::vector<cv::KeyPoint>& matched_query_keypoints, const cv::Mat& matched_query_descriptors,
                        const std::vector<cv::KeyPoint>& homography_train_keypoints, const cv::Mat& homography_train_descriptors,
                        const std::vector<cv::KeyPoint>& homography_query_keypoints, const cv::Mat& homography_query_descriptors,
                        const std::vector<cv::DMatch>& matches, const std::vector<cv::DMatch>& homography_matches,
                        const std::vector<std::pair<cv::Point3d, cv::Point3d>>& matched_points,
                        const std::vector<std::pair<cv::Point3d, cv::Point3d>>& homography_matched_points,
                        std::stringstream *errs)
//---------------------------------------------------------------------------------------------------------------------------------------
{
   dir = trim(dir);
   if (dir.empty())
      return false;
   else
   {
      filesystem::path directory(dir);
      if ( (filesystem::exists(directory)) && (! filesystem::is_directory(directory)) )
      {
         if (errs != nullptr)
            *errs << directory.string() << "exists but is not a directory";
         std::cerr << "yaml_write_matches: " << directory.string() << " exists but is not a directory";
         return false;
      }
      if (! filesystem::exists(directory))
         filesystem::create_directories(directory);
   }
   std::stringstream ss;
   ss << dir << "/" << name << "-matches.yaml";
   std::string yamlpath = ss.str();

   YAML::Emitter out;
   out.SetIndent(3);
   out << YAML::BeginMap;
   out << YAML::Key << "matched_train_keypoints" << YAML::Value << matched_train_keypoints;
   out << YAML::Key << "matched_train_descriptors" << YAML::Value << matched_train_descriptors;
   out << YAML::Key << "matched_query_keypoints" << YAML::Value << matched_query_keypoints;
   out << YAML::Key << "matched_query_descriptors" << YAML::Value << matched_query_descriptors;
   out << YAML::Key << "homography_train_keypoints" << YAML::Value << homography_train_keypoints;
   out << YAML::Key << "homography_train_descriptors" << YAML::Value << homography_train_descriptors;
   out << YAML::Key << "homography_query_keypoints" << YAML::Value << homography_query_keypoints;
   out << YAML::Key << "homography_query_descriptors" << YAML::Value << homography_query_descriptors;
   out << YAML::Key << "matches" << YAML::Value << matches;
   out << YAML::Key << "homography_matches" << YAML::Value << homography_matches;
   out << YAML::Key << "matched_points" << YAML::Value << matched_points;
   out << YAML::Key << "homography_matched_points" << YAML::Value << homography_matched_points;
   out << YAML::EndMap;
   std::ofstream of(yamlpath);
   of << out.c_str();
   return of.good();
}

bool yaml_write_3Dmatches(std::string dir, const std::string& name,
                          std::vector<std::tuple<cv::Point3d, cv::KeyPoint, cv::Mat>> threeDMatches)
//--------------------------------------------------------------------------------------------------
{
   dir = trim(dir);
   if (dir.empty())
      return false;
   else
   {
      filesystem::path directory(dir);
      if ( (filesystem::exists(directory)) && (! filesystem::is_directory(directory)) )
      {
         std::cerr << "yaml_write_3Dmatches: " << directory.string() << " exists but is not a directory";
         return false;
      }
      if (! filesystem::exists(directory))
         filesystem::create_directories(directory);
   }
   std::stringstream ss;
   ss << dir << "/" << name << "-matches3d.yaml";
   std::string yamlpath = ss.str();

   YAML::Emitter out;
   out.SetIndent(3);
   out << YAML::BeginMap;
   out << YAML::Key << "3d_matches" << YAML::Value;
   out << YAML::BeginSeq;
   for (const std::tuple<cv::Point3d, cv::KeyPoint, cv::Mat>& tt : threeDMatches)
   {
      out << YAML::BeginMap << YAML::Key << "Pt" << YAML::Value << std::get<0>(tt)
          << YAML::Key << "Kp" << YAML::Value << std::get<1>(tt)
          << YAML::Key << "Descriptor" << YAML::Value << std::get<2>(tt) <<  YAML::EndMap;
   }
   out << YAML::EndSeq;
   out << YAML::EndMap;
   std::ofstream of(yamlpath);
   of << out.c_str();
   return of.good();
}

#ifdef _MAIN_TEST_
void print_kp(const std::string name, const cv::KeyPoint& kp)
{
   std::cout << name << " = [ (" << kp.pt.x << ", " << kp.pt.y << ") size " << kp.size << " angle "
             << kp.angle << " response " << kp.response << " octave " << kp.octave << " class "
             << kp.class_id << "]" << std::endl;
}

int main(int argc, char **argv)
{
   cv::Mat m(4, 4, CV_32FC1);
   cv::randu(m, cv::Scalar(0), cv::Scalar(1));
   rapidjson::Document document;
   rapidjson::Document::AllocatorType& allocator = document.GetAllocator();
   document.SetObject();
   document.AddMember("m", encode_ocv_mat(document, allocator, m), allocator);
   cv::KeyPoint kp(1.0f, 0.5f, 50.0f, 90.0f, 1.0f, 2, 0);
   document.AddMember("k", encode_keypoint(document, allocator, kp), allocator);
   rapidjson::StringBuffer strbuf;
//   rapidjson::Writer<rapidjson::StringBuffer> writer(strbuf);
   rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(strbuf);
   document.Accept(writer);
   std::string json(strbuf.GetString());
   std::cout << json << std::endl;

   rapidjson::Document document2;
   rapidjson::Document::AllocatorType& allocator2 = document2.GetAllocator();
   document2.Parse(json.c_str());
   cv::Mat m2;
   cv::KeyPoint kp2;
   decode_ocv_mat(document2, "m", m2);
   std::cout << m << std::endl << "-------------" << std::endl; print_kp("kp", kp);
   std::cout << "==================" << std::endl << m2 << "--------------" << std::endl;
   decode_keypoint(document2, "k", kp2);
   print_kp("kp2", kp2);
}
#endif