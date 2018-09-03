#ifndef TRAINER_HULL_H
#define TRAINER_HULL_H

#include <opencv2/core/types.hpp>

int get_convex_hull(std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors, std::vector <cv::Point2f> &hull_points,
                    std::vector<cv::KeyPoint>& hull_keypoints, cv::Mat &hull_descriptors, cv::Point2f &centroid,
                    bool is_ref);
void show_convex_hull(const std::string win_name, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);

#endif //TRAINER_HULL_H
