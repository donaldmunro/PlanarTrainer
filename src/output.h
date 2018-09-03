#ifndef TRAINER_OUTPUT_H
#define TRAINER_OUTPUT_H

#include <opencv2/core/core.hpp>

#include "ImageMeta.h"

bool json_write(const cv::Mat& img, const cv::Mat& imgRGB, const std::vector<cv::KeyPoint>& k, const cv::Mat& d,
                const std::vector<cv::KeyPoint>& hull_keypoints, const cv::Mat& hull_descriptors, const cv::Point2f& centroid,
//              const std::vector<barycentric_info>& barycentrics, int max_triangles, float min_area_perc,
                const std::string& detector_name, const std::string& descriptor_name,
                int id, int rid, const std::string& name, std::string dir,
                double latitude, double longitude, float altitude, double depth, const cv::Rect2f& bb, const cv::RotatedRect& rbb,
                const imeta::ImageMeta& image_meta);

bool yaml_write(const cv::Mat& img, const cv::Mat& imgRGB, const std::vector<cv::KeyPoint>& k, const cv::Mat& d,
                const std::vector<cv::KeyPoint>& hull_keypoints, const cv::Mat& hull_descriptors, const cv::Point2f& centroid,
//              const std::vector<barycentric_info>& barycentrics, int max_triangles, float min_area_perc,
                const std::string& detector_name, const std::string& descriptor_name,
                int id, int rid, const std::string& name, std::string dir,
                double latitude, double longitude, float altitude, double depth, const cv::Rect2f& bb, const cv::RotatedRect& rbb,
                const imeta::ImageMeta& image_meta);

bool ocv_write(const cv::Mat& img, const cv::Mat& imgRGB, const std::vector<cv::KeyPoint>& k, const cv::Mat& d,
               const std::vector<cv::KeyPoint>& hull_keypoints, const cv::Mat& hull_descriptors, const cv::Point2f& centroid,
//              const std::vector<barycentric_info>& barycentrics, int max_triangles, float min_area_perc,
               const std::string& detector_name, const std::string& descriptor_name,
               int id, int rid, const std::string& name, std::string dir,
               double latitude, double longitude, float altitude, double depth, const cv::Rect2f& bb, const cv::RotatedRect& rbb,
               const imeta::ImageMeta& image_meta);

bool ocv_write_matches(std::string dir, const std::string& name, const std::vector<cv::KeyPoint>& matched_train_keypoints,
                       const cv::Mat& matched_train_descriptors,
                       const std::vector<cv::KeyPoint>& matched_query_keypoints, const cv::Mat& matched_query_descriptors,
                       const std::vector<cv::KeyPoint>& homography_train_keypoints, const cv::Mat& homography_train_descriptors,
                       const std::vector<cv::KeyPoint>& homography_query_keypoints, const cv::Mat& homography_query_descriptors,
                       const std::vector<cv::DMatch>& matches, const std::vector<cv::DMatch>& homography_matches,
                       std::stringstream *errs = nullptr);

bool json_write_matches(std::string dir, const std::string& name,
                        const std::vector<cv::KeyPoint>& matched_train_keypoints, const cv::Mat& matched_train_descriptors,
                        const std::vector<cv::KeyPoint>& matched_query_keypoints, const cv::Mat& matched_query_descriptors,
                        const std::vector<cv::KeyPoint>& homography_train_keypoints, const cv::Mat& homography_train_descriptors,
                        const std::vector<cv::KeyPoint>& homography_query_keypoints, const cv::Mat& homography_query_descriptors,
                        const std::vector<cv::DMatch>& matches, const std::vector<cv::DMatch>& homography_matches,
                        std::stringstream *errs = nullptr);

bool yaml_write_matches(std::string dir, const std::string& name,
                        const std::vector<cv::KeyPoint>& matched_train_keypoints, const cv::Mat& matched_train_descriptors,
                        const std::vector<cv::KeyPoint>& matched_query_keypoints, const cv::Mat& matched_query_descriptors,
                        const std::vector<cv::KeyPoint>& homography_train_keypoints, const cv::Mat& homography_train_descriptors,
                        const std::vector<cv::KeyPoint>& homography_query_keypoints, const cv::Mat& homography_query_descriptors,
                        const std::vector<cv::DMatch>& matches, const std::vector<cv::DMatch>& homography_matches,
                        const std::vector<std::pair<cv::Point3d, cv::Point3d>>& matched_points,
                        const std::vector<std::pair<cv::Point3d, cv::Point3d>>& homography_matched_points,
                        std::stringstream *errs =nullptr);

bool yaml_write_3Dmatches(std::string dir, const std::string& name,
                          std::vector<std::tuple<cv::Point3d, cv::KeyPoint, cv::Mat>> threeDMatches);


#endif