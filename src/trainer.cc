#include <stdio.h>
#include <math.h>
#include <sys/stat.h>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <functional>
#include <algorithm>
#include <map>
#include <random>

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
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "opencv2/opencv_modules.hpp"
#ifdef HAVE_OPENCV_XFEATURES2D
#include <opencv2/xfeatures2d.hpp>
#endif

#include <QApplication>
#include <QCommandLineParser>
#include <QMessageBox>
#include <QtWidgets>
#include <QInputDialog>
#include <QTimer>

#define GLM_ENABLE_EXPERIMENTAL

#include "util.h"
#include "hull.h"
#include "ImageMeta.h"
#include "output.h"
#include "ImageWindow.hh"
#include "Calibration.hh"

#define DEBUG

int main(int argc, char **argv)
//-----------------------------
{
   QApplication a(argc, argv);
   std::stringstream errs;
   QApplication::setApplicationName("Trainer");
   QApplication::setApplicationVersion("0.1");
   QCommandLineParser parser;
   parser.setApplicationDescription("Trainer");
   parser.addHelpOption();
   parser.addVersionOption();
   parser.addOption(QCommandLineOption("t", "Train image", "train-image", ""));
   parser.addOption(QCommandLineOption("q", "Query image", "query-image", ""));
   parser.addOption(QCommandLineOption("c",
                                       "Camera Calibration. Either -c fx,fy,ox,oy (no spaces or surrounded by quotes)\n"
                                       "or -c file where file contains the same comma delimited form\n"
                                       "or a YAML file with a .yaml extension containing for eg\n"
                                       "fx: 123,45\n"
                                       "fy: 456.78 ...", "calibration", ""));
   parser.process(a);
   std::string image_filename = parser.value("t").toStdString();
   std::string query_image_filename = parser.value("q").toStdString();
   std::string calibration_arg = parser.value("c").toStdString();
   std::unique_ptr<Calibration> calibration;
   if (calibration_arg.empty())
      std::cout << "WARNING: No camera calibration specified. Pose option not supported unless calibration read in UI file menu.\n";
   else
   {
      filesystem::path calibration_file(calibration_arg);
      if (filesystem::exists(calibration_file))
      {
         calibration.reset(new Calibration);
         if (! calibration->open(calibration_file.string(), &errs))
         {
            std::cerr << "Calibration file error: " << errs.str() << std::endl;
            calibration.reset(nullptr);
            return 1;
         }
      }
      else
      {
         calibration.reset(new Calibration);
         if (! calibration->set(calibration_arg, &errs))
         {
            std::cerr << "Calibration argument error: " << errs.str() << std::endl;
            return 1;
         }
      }
   }

   ImageWindow imgwin(&a);
   if (calibration)
      imgwin.set_calibration(calibration);
   imgwin.showMaximized();
   if (! image_filename.empty())
   {
      filesystem::path imagefile(image_filename);
      if (filesystem::is_regular_file(imagefile))
      {
         if (!imgwin.load_train_img(image_filename, &errs))
         {
            std::cerr << "WARNING: Command line image file " << image_filename << " not loaded (" << errs.str() << std::endl;
            return 1;
         }
      }
      else
      {
         std::cerr << "WARNING: Command line image file " << image_filename << " not found" << std::endl;
         return 1;
      }
   }
   if (! query_image_filename.empty())
   {
      std::stringstream errs;
      filesystem::path imagefile(query_image_filename);
      if (filesystem::is_regular_file(imagefile))
      {
         if (!imgwin.load_query_img(query_image_filename, &errs))
            std::cerr << "WARNING: Command line image file " << query_image_filename << " not loaded (" << errs.str() << std::endl;
      }
      else
         std::cerr << "WARNING: Command line image file " << query_image_filename << " not found" << std::endl;
   }
   a.exec();
}

cv::Rect adjusted_region(const cv::Rect& region)
//--------------------------------
{
   cv::Rect r(region);
   if (r.width < 0)
   {
      r.x += r.width;
      r.width *= -1;
   }
   if (r.height < 0)
   {
      r.y += r.height;
      r.height *= -1;
   }
   return r;
}

void msg(const char *psz)
//-------------------------------
{
   QMessageBox msgBox;
   msgBox.setText(QString(psz));
   msgBox.exec();
}

void msg(const std::stringstream &errs)
//-------------------------------
{
   QMessageBox msgBox;
   msgBox.setText(QString(errs.str().c_str()));
   msgBox.exec();
}

/*
inline void counterclockwise_sort(std::vector<cv::Point2f>& vertices)
//-------------------------------------------------------------------
{
   cv::Point2f pt1 = vertices[0], pt2 = vertices[1], pt3 = vertices[2];
   cv::Point2f centroid(pt1.x, pt1.y);
   centroid.x += pt2.x; centroid.y += pt2.y;
   centroid.x += pt3.x; centroid.y += pt3.y;
   centroid.x /= 3; centroid.y /= 3;
   const float angle1 = atan2f(pt1.y - centroid.y, pt1.x - centroid.x);
   const float angle2 = atan2f(pt2.y - centroid.y, pt2.x - centroid.x);
   const float angle3 = atan2f(pt3.y - centroid.y, pt3.x - centroid.x);
   if (angle1 > angle2)
   {
      cv::Point2f tmppt = vertices[0];
      vertices[0] = vertices[1];
      vertices[1] = tmppt;
   }
   if (angle1 > angle3)
   {
      cv::Point2f tmppt = vertices[0];
      vertices[0] = vertices[2];
      vertices[2] = tmppt;
   }
   if (angle2 > angle3)
   {
      cv::Point2f tmppt = vertices[1];
      vertices[1] = vertices[2];
      vertices[2] = tmppt;
   }
}

inline bool clockwise_less(cv::Point2f a, cv::Point2f b, cv::Point2f center)
//--------------------------------------------------------------------------
{
   if (a.x - center.x >= 0 && b.x - center.x < 0)
      return true;
   if (a.x - center.x < 0 && b.x - center.x >= 0)
      return false;
   if (a.x - center.x == 0 && b.x - center.x == 0) {
      if (a.y - center.y >= 0 || b.y - center.y >= 0)
         return a.y > b.y;
      return b.y > a.y;
   }

   // compute the cross product of vectors (center -> a) x (center -> b)
   float det = (a.x - center.x) * (b.y - center.y) - (b.x - center.x) * (a.y - center.y);
   if (det < 0)
      return true;
   if (det > 0)
      return false;

   // points a and b are on the same line from the center
   // check which point is closer to the center
   float d1 = (a.x - center.x) * (a.x - center.x) + (a.y - center.y) * (a.y - center.y);
   float d2 = (b.x - center.x) * (b.x - center.x) + (b.y - center.y) * (b.y - center.y);
   return d1 > d2;
}

bool find_normal(Calibration *calibration_values, imeta::ImageMeta& image_meta,
                 const std::vector<cv::KeyPoint>& hull_kpts, const std::vector<cv::KeyPoint>& train_kpts,
                 cv::Mat& normal, cv::Mat& rotated_normal)
//-------------------------------------------------------------------------------------------------------
{
   std::stringstream ss;
   std::vector<cv::KeyPoint> top_keypoints((hull_kpts.size() >= 9) ? hull_kpts : train_kpts);
   cv::KeyPointsFilter::removeDuplicated(top_keypoints);
   cv::KeyPointsFilter::retainBest(top_keypoints, 60);
   std::vector<cv::Point2f> pts;
   cv::KeyPoint::convert(top_keypoints, pts);
   std::vector<cv::Vec6f> triangle_vertices;

   cv::Mat R, RM;
   getRotationMatrixFromMat<double>(R, image_meta.rotation_vector, 4, CV_64FC1);
   //R = cv::Mat::eye(4, 4, CV_64FC1);
   unsigned X, Y;
   switch (image_meta.device_rotation)
   {
      case 90: X = AXIS_Y; Y = AXIS_MINUS_X; break;
      case 180: X = AXIS_MINUS_X; Y = AXIS_MINUS_Y; break;
      case 270: X = AXIS_MINUS_Y; Y = AXIS_X; break;
      default: X = AXIS_X; Y = AXIS_Y; break;
   }
   remapCoordinateSystem<double>(R, X, Y, RM, CV_64FC1);
   cv::Mat RMI = RM.t();
   std::cout << RM << std::endl;
   normal = cv::Mat::zeros(1, 3, CV_64FC1);
   rotated_normal = cv::Mat::zeros(1, 3, CV_64FC1);
   double *an = normal.ptr<double>(0);
   double *ran = rotated_normal.ptr<double>(0);
   int c = 0;
   ss << "clf;" << std::endl;
   std::default_random_engine generator;
//   std::uniform_int_distribution<size_t> distribution(0, n);
   constexpr double min_area = 50;
   std::vector<cv::Point2f> remaining_pts;
   while (pts.size() >= 3)
   {
      std::vector<cv::Point2f> vertices;
      std::vector<size_t> indices;
      size_t n = pts.size();
      for (size_t i=0; i<n; i++)
         indices.push_back(i);
      std::shuffle(indices.begin(), indices.end(), generator);
      auto it = indices.begin();
      int retries = 0;
      while ( it != indices.end())
      {
         vertices.clear();
         for (int i=0; i<3; i++)
         {
            size_t j = *it;
            vertices.push_back(pts[j]);
            if (++it == indices.end())
               break;
         }
         if (vertices.size() == 3)
         {
            //counterclockwise_sort(vertices);
            cv::Point2f& pt1 = vertices[0], &pt2 = vertices[1], &pt3 = vertices[2];
            cv::Point2f centroid(pt1.x, pt1.y);
            centroid.x += pt2.x; centroid.y += pt2.y;
            centroid.x += pt3.x; centroid.y += pt3.y;
            centroid.x /= 3; centroid.y /= 3;
            std::sort(std::begin(vertices), std::end(vertices),
                      [centroid](const cv::Point2f &lhs, const cv::Point2f &rhs) -> bool
                            //----------------------------------------------------------------------------------
                      {
                         return (clockwise_less(lhs, rhs, centroid));
                      });
            cv::Point2d v1 = vertices[0], v2 = vertices[1], v3 = vertices[2];
            double a[] { v1.x, v1.y, 1.0f, v2.x, v2.y, 1.0f, v3.x, v3.y, 1.0f};
            cv::Mat A(3, 3, CV_64FC1, static_cast<void *>(a));
            double area = 0.5 * fabs(cv::determinant(A));
            if (area >= min_area)
            {
               cv::Mat p1 = calibration_values->image2camera(v1, 4).t();
               cv::Mat p2 = calibration_values->image2camera(v2, 4).t();
               cv::Mat p3 = calibration_values->image2camera(v3, 4).t();
               cv::Mat vec1 = p1.rowRange(0, 3) - p2.rowRange(0, 3);
               cv::Mat vec2 = p3.rowRange(0, 3) - p2.rowRange(0, 3);
               cv::Vec3d n = vec1.cross(vec2);
               double mag = sqrt(n.ddot(n));
               n[0] /= mag; n[1] /= mag; n[2] /= mag;
               std::cout << v1 << " " << v2 << " " << v3 << " " << area << " *** " << n << std::endl;
               an[0] += n[0]; an[1] += n[1]; an[2] += n[2];

               p1 = RMI * p1;
               p2 = RMI * p2;
               p3 = RMI * p3;
               vec1 = p1.rowRange(0, 3) - p2.rowRange(0, 3);
               vec2 = p3.rowRange(0, 3) - p2.rowRange(0, 3);
               n = vec1.cross(vec2);
               mag = sqrt(n.ddot(n));
               n[0] /= mag; n[1] /= mag; n[2] /= mag;
               std::cout << v1 << " " << v2 << " " << v3 << " " << area << " *** " << n << std::endl;
               ran[0] += n[0]; ran[1] += n[1]; ran[2] += n[2];

               c++;
            }
            else
            {
               remaining_pts.push_back(v1);
               remaining_pts.push_back(v2);
               remaining_pts.push_back(v3);
               std::cout << v1 << " " << v2 << " " << v3 << " " << area << " reject" << std::endl;
            }
         }
      }
      pts.clear();
      if ( (c < 10) && (remaining_pts.size() > 0) && (retries++ < 3) )
         pts = std::move(remaining_pts);
   }

//   cv::Subdiv2D triangulator(bb);
//   triangulator.insert(pts);
//   triangulator.getTriangleList(triangle_vertices);
//   for (cv::Vec6f triangle : triangle_vertices)
//   {
//   }
   if (c == 0)
      return false;
   double cf = static_cast<double>(c);
   an[0] /= cf; an[1] /= cf; an[2] /= cf;
   ran[0] /= cf; ran[1] /= cf; ran[2] /= cf;
   std::cout << normal << " " << rotated_normal << std::endl;
   return true;
}
*/

//void barycentric_calc(std::vector<cv::KeyPoint>& keypoints, cv::Point2f centroid, bool is_hull_only, float avg_response,
//                      cv::Mat trainImgRGB, std::vector<barycentric_info_t> &barycentrics)
////---------------------------------------------------------------------------------------------------------------------
//{
//   std::vector<cv::KeyPoint> triangle_candidates;
//   for (int i=0; i< static_cast<int>(keypoints.size()); i++)
//   {
//      KeyPointExtra_t kpe;
//      cv::KeyPoint& kp = keypoints[i];
//      if (kp.response < avg_response)
//         continue;
//      kpe.class_id = static_cast<uint32_t>(kp.class_id);
//      if ( (is_hull_only) && (! kpe.is_hull) )
//         continue;
//      triangle_candidates.push_back(kp);
//   }
//   std::sort(triangle_candidates.begin(), triangle_candidates.end(),
//             [](const cv::KeyPoint &kp1, const cv::KeyPoint &kp2) -> bool
//                   //----------------------------------------------------------
//             {
//                return kp2.response < kp1.response;
//             });
//   std::vector<std::vector<cv::KeyPoint>> triangles;
//   std::unique_ptr<Triangulator> triangulator(new OrderedTriangulator());
//   triangulator->triangulate(triangle_candidates, triangles);
//#ifdef DEBUG
//   cv::Mat debug_img;
//   trainImgRGB.copyTo(debug_img);
//   int tn = 0;
//#endif
//   for (std::vector<cv::KeyPoint> triangle : triangles)
//   {
//#ifdef DEBUG
//      draw_triangle(debug_img, triangle);
//      cv::Mat one_triangleim; trainImgRGB.copyTo(one_triangleim);
//      draw_triangle(one_triangleim, triangle);
//      std::stringstream ss;
//      ss << "triangle-" << std::setfill('0') << std::setw(3) << tn++ << ".png";
//      cv::imwrite(ss.str(), one_triangleim);
//#endif
//      barycentric_info info;
//      std::vector<float> coordinates;
//      if (barycentric(triangle, centroid, coordinates))
//      {
//         float data[]{triangle[0].pt.x, triangle[0].pt.y, 1.0f,
//                      triangle[1].pt.x, triangle[1].pt.y, 1.0f,
//                      triangle[2].pt.x, triangle[2].pt.y, 1.0f};
//         cv::Mat A(3, 3, CV_32FC1, static_cast<void *>(data));
//         info.area = 0.5f * fabs(static_cast<float>(cv::determinant(A)));
//         for (int i=0; i<3; i++)
//         {
//            cv::KeyPoint &kp = triangle[i];
//            KeyPointExtra_t kpe;
//            kpe.class_id = (uint32_t) kp.class_id;
//            info.vertex_keypointno[i] = kpe.no;
//            info.coordinates[i] = coordinates[i];
//         }
//         barycentrics.push_back(info);
//      }
//   }
//
////      cv::imshow("win", debug_img); cv::waitKey(0);
//   cv::imwrite("triangles.png", debug_img);
//}

//int combinations(const std::vector<cv::KeyPoint>& keypoints, int r,
//                 std::function<bool (std::vector<cv::KeyPoint>)> fn_next_combination);
//void xratio(const std::vector<cv::KeyPoint>& keypoints, const cv::Point2f& centroid,
//            std::vector<xratio_info_t>& xratios);
//
//int combinations(const std::vector<cv::KeyPoint>& v, int r,
//                 std::function<bool (const std::vector<cv::KeyPoint>)> fn_next_combination)
////------------------------------------------------------------------------------------------------------------------------------
//{
//   int n = static_cast<int>(v.size());
//   std::vector<bool> b(n);
//   std::fill(b.begin(), b.begin() + r, true);
//   do
//   {
//      std::vector<cv::KeyPoint> combination;
//      for (int i = 0; i < n; ++i)
//      {
//         if (b[i])
//            combination.push_back(v[i]);
//      }
//      if (! fn_next_combination(combination))
//         break;
//   } while (std::prev_permutation(b.begin(), b.end()));
//   return 0;
//}
//
//void xratio(const std::vector<cv::KeyPoint>& keypoints, const cv::Point2f& centroid,
//            std::vector<xratio_info_t>& xratios)
////----------------------------------------------------------------------------------
//{
//   std::function<bool (std::vector<cv::KeyPoint>)> fn =
//         [&centroid, &xratios](std::vector<cv::KeyPoint> vertices) -> bool
//               //------------------------------------------------------------------------
//         {
//            int  i = 0;
//            float data[] {vertices[0].pt.x, vertices[0].pt.y, 1.0f,
//                          vertices[1].pt.x, vertices[1].pt.y, 1.0f,
//                          vertices[2].pt.x, vertices[2].pt.y, 1.0f};
//            cv::Mat A(3, 3, CV_32FC1, static_cast<void *>(data));
//            float area = 0.5f*fabs(static_cast<float>(cv::determinant(A)));
//            if (area > 0.00001)
//               return true;
//            data[6] = centroid.x;
//            data[7] = centroid.y;
//            A = cv::Mat(3, 3, CV_32FC1, static_cast<void *>(data));
//            area = 0.5f*fabs(static_cast<float>(cv::determinant(A)));
//            if (area > 0.00001)
//               return true;
//            KeyPointExtra_t kpe;
//            kpe.no = 0;
//            kpe.is_hull = kpe.is_selected = 0;
//            kpe.imgIdx = 255;
//            cv::KeyPoint kpc(centroid, 0, -1, -1);
//            kpc.class_id = kpe.class_id;
//            vertices.push_back(kpc);
//            std::sort(vertices.begin(), vertices.end(),
//                      [](const cv::KeyPoint &kp1, const cv::KeyPoint &kp2) -> bool
//                            //--------------------------------------------------------
//                      {
//                         float dx = kp1.pt.x, dy = kp1.pt.y;
//                         float d1 = dx*dx + dy*dy;
//                         dx = kp2.pt.x; dy = kp2.pt.y;
//                         return d1 < (dx*dx + dy*dy);
//                      });
//            cv::Point2f &p1 = vertices[0].pt, &p2 = vertices[1].pt,
//                  &p3 = vertices[2].pt, &p4 = vertices[3].pt;
//
//            xratio_info info;
//            info.xratio = (euclidean2(p1, p3) * euclidean2(p2, p4)) / euclidean2(p1, p4) * euclidean2(p2, p3);
//            i = 0;
//            for (cv::KeyPoint kp : vertices)
//            {
//               kpe.class_id = static_cast<uint32_t>(kp.class_id);
//               if (kpe.imgIdx == 255)
//                  info.centre_index = i;
//               info.vertex_keypointno[i++] = kpe.no;
//            }
//         };
//   combinations(keypoints, 3, fn);
//}
