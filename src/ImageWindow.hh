#ifndef _IMAGE_WINDOW_HH_
#define _IMAGE_WINDOW_HH_
#include <memory>
#include <functional>
#include <array>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <limits>
#include <exception>
#ifdef FILESYSTEM_EXPERIMENTAL
#include <experimental/filesystem>
namespace filesystem = std::experimental::filesystem;
#elif defined(STD_FILESYSTEM)
#include <filesystem>
namespace filesystem = std::filesystem;
#else
#include "filesystem.hpp"
#endif

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#ifdef HAVE_OPENCV_XFEATURES2D
#include <opencv2/xfeatures2d.hpp>
#endif

#include <QApplication>
#include <QTimer>
#include <QMainWindow>
#include <QImage>
#include <QLabel>
#include <QAction>
#include <QMenu>
#include <QVBoxLayout>
#include <QWidget>
#include <QTabWidget>
#include <QPoint>
#include <QRect>
#include <QRadioButton>
#include <QCheckBox>
#include <QButtonGroup>
#include <QListWidget>
#include <QComboBox>
#include <QLineEdit>
#include <QPushButton>
#include <QFileDialog>
#include <QMessageBox>
#include <QDialog>
#include <QStackedLayout>

#include "CVQtImage.h"
#include "CVQtScrollableImage.h"
#include "typ.h"
#include "ImageMeta.h"
#include "KeypointFlann.hh"
#include "Calibration.hh"
#include "Dialogs.h"
#include "pose/Pose.hh"

#define MAX_POSE_RESULTS 5

class ImageWindow : public QMainWindow
//=====================================
{  Q_OBJECT

public:
   explicit ImageWindow(QApplication *a, QWidget* parent = nullptr);
   virtual ~ImageWindow() {  }

   bool load_train_img(std::string imagefile, std::stringstream *errs = nullptr);
   bool load_query_img(std::string imagefile, std::stringstream *errs = nullptr);
   void set_calibration(std::unique_ptr<Calibration>& calibration_ptr) { calibration = std::move(calibration_ptr); }
//   const DetectorInfo& detector() const { return detector_info; };
   std::string save_dialog_name() { return save_dialog_result.toStdString(); }
   bool yes_no() { return yes_no_result; }

protected:
   virtual bool create_detector(std::string detectorName, cv::Ptr<cv::Feature2D>& detector);
   virtual void drawKeypoint(cv::InputOutputArray img, const cv::KeyPoint& p, const cv::Scalar& color, int flags);

private slots:
   void tab_selected();
   void open_train_image();
   void open_query_image();
   void open_calibration_file();
   void exit() { app->quit(); /*std::exit(1);*/ }
   void save();
   void load();
   void save_matches();
   void load_matches();
   void load_3dmatches();
   void on_clear_selection();
   void on_selected_from_matched();
   void on_selected_from_homography();
   void on_delete_unselected();

   void on_detect();
   void on_match();
   void on_pose();
   void on_selected_color_change();
   void on_unselected_color_change();
   void pose_method_selected();

   void onOrbSelected() { is_bit_descriptor  = true; setNorms(); }
   void onSiftSelected() { is_bit_descriptor  = false; setNorms();  }
   void onSurfSelected() { is_bit_descriptor  = false; setNorms(); }
   void onAkazeSelected() { is_bit_descriptor  = true; setNorms(); }

   void setNorms();

   void onBriskSelected() { is_bit_descriptor  = true; setNorms(); }

private:
   QTabWidget tabs;
   QVBoxLayout* panel_layout;
   QMenu* file_menu;
   QAction *open_train_img_action, *open_query_img_action, *open_calibration_action, *save_action, *load_action,
           *save_matches_action, *load_matches_action, *load_3dmatches_action, *exit_action;
   QMenu* action_menu;
   QAction *detect_action, *match_action, *clear_selected_action, *selected_from_matched_action, *selected_from_homography_action,
           *delete_unselected_action;
   QApplication *app;
   QString save_dialog_result;
   bool yes_no_result = false;
   bool is_bit_descriptor = true;

   QButtonGroup featureDetectorsRadioGroup, poseSourceRadioGroup;
   QListWidget *listOrbWTA, *listOrbScore, *listAkazeDescriptors, *listAkazeDiffusivity, *listMatcherType,
               *listMatcherNorm, *listHomographyMethod, *listPoseMethod;
   QLineEdit *editOrbNoFeatures, *editOrbScale, *editOrbPatchSize,
             *editSiftNoFeatures, *editSiftOctaveLayers, *editSiftContrast, *editSiftEdge, *editSiftSigma,
             *editSurfThreshold, *editSurfOctaves, *editSurfOctaveLayers,
             *editAkazeThreshold, *editAkazeOctaves, *editAkazeOctaveLayers,
             *editBriskThreshold, *editBriskOctaves, *editBriskScale, *editBest,
             *editHomographyReprojErr, *editPoseRANSACError;
   QCheckBox *chkSurfExtended, *chkPoseUseRANSAC;
   QRadioButton *orbRadio, *siftRadio, *surfRadio, *akazeRadio, *briskRadio, *poseFromHomography,
                *poseFromMatched;
   QPushButton *detectButton, *matchButton, *selectedColorButton, *unselectedColorButton, *poseButton;
   CVQtScrollableImage* train_image_holder, *query_image_holder;
   QStackedLayout *stackedPoseLayout;
   QComboBox *posePageComboBox;
   QWidget* pose_widget;
   CVQtImage *match_image_holder;
   size_t no_results = 1;
   std::vector<QLabel *> translation_labels, rotation_labels, error_labels;
   std::vector<QLabel *> pose_image_labels;
   std::vector<QScrollArea *> pose_scrolls;

   cv::Mat pre_detect_train_image, pre_detect_train_image_bw, pre_detect_query_image, pre_detect_query_image_bw;
   imeta::ImageMeta train_image_meta, query_image_meta;
   std::vector<cv::KeyPoint> all_train_keypoints, all_query_keypoints;
   cv::Mat all_train_descriptors, all_query_descriptors;
   std::string feature_detector_name, descriptor_extractor_name;
   KeyPointFlannSource index_points;
   ThreeDFlannSource index_points3d;
   std::unique_ptr<keypoint_kd_tree_t> index;
   std::unique_ptr<threeD_kd_tree_t> index3d;
   std::vector<features_t> region_features;
   DetectorInfo detector_info;
   cv::Ptr<cv::Feature2D> detector;
   std::vector<cv::KeyPoint> matched_train_keypoints, matched_query_keypoints,
                             homography_train_keypoints, homography_query_keypoints;
   cv::Mat matched_train_descriptors, matched_query_descriptors, homography_train_descriptors,
           homography_query_descriptors;
   std::vector<cv::DMatch> matches, homography_matches;
   std::vector<std::pair<cv::Point3d, cv::Point3d>> matched_points, homography_matched_points;
   std::vector<std::tuple<cv::Point3d, cv::KeyPoint, cv::Mat>> threeD_matches;
   std::vector<std::pair<cv::Point3d, cv::Point2d>> loaded_threeD_matches;

   std::unique_ptr<Calibration> calibration;

   QColor selected_colour{255, 0, 0}, unselected_colour{0, 255, 0};
   const cv::Scalar red = cv::Scalar(0, 0, 255), green = cv::Scalar(0, 255, 0), yellow = cv::Scalar(0, 255, 255);
   ThreeDCoordsDialog* enter_3D_coords;
   std::tuple<cv::Point3d, cv::KeyPoint, cv::Mat>* edit_3D_tuple = nullptr;

   void on_left_click(cv::Point2f& pt, bool is_shift, bool is_ctrl, bool is_alt);
   void on_right_click(cv::Point2f& pt, bool is_shift, bool is_ctrl, bool is_alt);
   void on_region_selected(cv::Mat& roi, cv::Rect& roirect, bool is_shift, bool is_ctrl, bool is_alt);
   void on_accept_3d_point(cv::KeyPoint& keypoint, cv::Mat& descriptor, cv::Point3d& pt3d);

   void setup_features_control(QLayout *layout);
   QWidget* pose_layout();
   static bool load_image(const std::string imagepath, cv::Mat& pre_color_img, cv::Mat& pre_mono_image,
                          CVQtScrollableImage* image_holder, imeta::ImageMeta& image_meta,
                          std::stringstream* errs =nullptr, std::string imagemeta_path ="");
   void drawKeypoints(const cv::InputArray& image, std::vector<cv::KeyPoint>& keypoints,
                      cv::InputOutputArray& outImage, int flags =cv::DrawMatchesFlags::DEFAULT,
                      std::vector<cv::KeyPoint*>* selected_pts = nullptr);
   bool closest_keypoint(float x, float y, float radius, cv::KeyPoint*& kp, float& distance);
   bool closest_3D_point(float x, float y, float radius, std::tuple<cv::Point3d, cv::KeyPoint, cv::Mat>*& tt,
                         float& distance);
   void response_stats(std::vector<cv::KeyPoint>& keypoints, double& min_response, double &max_response,
                       double& response_mean, double& response_variance, double& response_deviation);
   void set_pose_result(size_t no, Eigen::Vector3d& T, Eigen::Quaterniond& Q, double maxError, double meanError,
                        cv::Mat* image =nullptr);
   QWidget* addPoseResultWidget();
   size_t load_3D_points(std::vector<std::pair<cv::Point3d, cv::Point3d>>& pts);

   inline void msgbox(QString s)
   {
      QMessageBox msgBox;
      msgBox.setText(s);
      msgBox.exec();
   }
   inline void yesnobox(QString title, QString msg)
   {
      QMessageBox::StandardButton reply;
      reply = QMessageBox::question(this, title, msg, QMessageBox::Yes | QMessageBox::No);
      yes_no_result =  (reply == QMessageBox::Yes);
   }

   void select_from(std::vector<cv::KeyPoint>& all_keypoints, std::vector<cv::KeyPoint>& keypoints);
   void set_pose_image(const cv::Mat& image, QLabel* imageLabel);
};

#endif // MAINWINDOW_HH
