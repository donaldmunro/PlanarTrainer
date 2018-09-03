#ifndef TRAINER_IMAGEMETA_H
#define TRAINER_IMAGEMETA_H

#include <string>
#include <sstream>
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

#include <yaml-cpp/yaml.h>

#include "Calibration.hh"

namespace imeta
{

   template<typename T>
   void yaml_mat(int rows, int cols, YAML::Node& n, cv::Mat& m, int mattype =CV_64F )
   //---------------------------------------------------------
   {
      m = cv::Mat::zeros(1, 3, mattype);
      for (int row =0; row<rows; row++)
         for (int col =0; col<cols; col++)
            m.at<T>(row, col) = n[col + row*cols].as<T>();

   }

   class ImageMeta
   //=============
   {
   public:
      ImageMeta() = default;
      explicit ImageMeta(std::string filename);
      bool open(const filesystem::path& metafile);
      inline explicit operator bool() const { return is_good; }
      std::string messages() { return log.str(); }

      std::stringstream log;
      int device_rotation =-1;
      float latitude =std::numeric_limits<float>::quiet_NaN();
      float longitude =std::numeric_limits<float>::quiet_NaN();
      float altitude =std::numeric_limits<float>::quiet_NaN(), z_deviation =0, depth =0;
      bool has_location = false;
      cv::Mat deviceGravity, correctedGravity, rotation_vector;
      filesystem::path metafile;

      static bool is_opencv_yaml(filesystem::path metafile);

      std::unique_ptr<Calibration> calibration;

   private:
      filesystem::path get_metafile(filesystem::path metafile);
      bool is_good = false;

      bool open_yaml(filesystem::path metafile);


   };
}

#endif //TRAINER_IMAGEMETA_H
