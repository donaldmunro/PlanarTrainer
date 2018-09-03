#include "ImageWindow.hh"

#include <memory>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <regex>
#include <cmath>
#include <algorithm>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/imgcodecs.hpp>

#include <QList>
#include <QAction>
#include <QGuiApplication>
#include <QScreen>
#include <QFont>
#include <QFontMetrics>
#include <QMenuBar>
#include <QMenu>
#include <QTimer>
#include <QIntValidator>
#include <QDoubleValidator>
#include <QImageReader>
#include <QFileDialog>
#include <QStandardPaths>
#include <QLayout>
#include <QImageWriter>
#include <QLabel>
#include <QAbstractItemView>
#include <QFrame>
#include <QSplitter>
#include <QDebug>
#include <QDialogButtonBox>
#include <QtWidgets/QColorDialog>
#include <QtWidgets/QFormLayout>
#include <pose/Ransac.hh>

#include "ocv.h"
#include "util.h"
#include "output.h"
#include "hull.h"
#include "pose/math.hh"
#include "pose/GravityPose.hh"
#include "pose/OtherPose.hh"

#define FEATURE_CONTROL_TAB 0
#define TRAIN_IMAGE_TAB 1
#define QUERY_IMAGE_TAB 2
#define MATCHES_TAB 3
#define POSE_TAB 4

inline QString listItem(const QListWidget *list)
//---------------------------------------
{
   QListWidgetItem *item = list->currentItem();
   if ( (item == nullptr) || (item->text().isEmpty()) )
   {
      const QList<QListWidgetItem *> &sel = list->selectedItems();
      if (sel.size() > 0)
         item = sel.at(0);
   }
   if ( (item == nullptr) || (item->text().isEmpty()) )
      item = list->item(0);
   if (item == nullptr)
      return QString("");
   return item->text();
}

inline bool valInt(QLineEdit *edit, int& v, const std::string& errmsg)
//--------------------------------------------------------------------
{
   bool ok;
   v = edit->text().trimmed().toInt(&ok);
   if (! ok)
   {
      std::stringstream errs;
      errs <<  errmsg << " (" << edit->text().toStdString() << ")";
      msg(errs);
      return false;
   }
   return true;
}

inline bool valFloat(QLineEdit *edit, float& v, const std::string& errmsg)
//-------------------------------------------------------------------------
{
   bool ok;
   v = edit->text().trimmed().toFloat(&ok);
   if (! ok)
   {
      std::stringstream errs;
      errs <<  errmsg << " (" <<  edit->text().toStdString() << ")";
      msg(errs);
      return false;
   }
   return true;
}

inline bool valDouble(QLineEdit *edit, double& v, const std::string& errmsg)
//-------------------------------------------------------------------------
{
   bool ok;
   v = edit->text().trimmed().toDouble(&ok);
   if (! ok)
   {
      std::stringstream errs;
      errs <<  errmsg << " (" <<  edit->text().toStdString() << ")";
      msg(errs);
      return false;
   }
   return true;
}

inline size_t find_keypoint(const std::vector<cv::KeyPoint>& keypoints, const cv::KeyPoint& the_keypoint)
//--------------------------------------------------------------------------------------------------
{
   const float x = the_keypoint.pt.x;
   const float y = the_keypoint.pt.y;
   const float s = the_keypoint.size;
   const float r = the_keypoint.response;
   const float epsilon = 0.00001f;
   size_t i = 0;
   for (const cv::KeyPoint& keypoint : keypoints)
   {
      const float xi = keypoint.pt.x;
      const float yi = keypoint.pt.y;
      const float si = keypoint.size;
      const float ri = keypoint.response;
      if ( (mut::near_zero(x - xi, epsilon)) && (mut::near_zero(y - yi, epsilon)) &&
            (mut::near_zero(s - si, epsilon)) && (mut::near_zero(r - ri, epsilon)) )
         return i;
      i++;
   }
   return std::numeric_limits<size_t>::max();
}

inline size_t find_image_point(const std::vector<std::pair<cv::Point3d, cv::Point3d>>& points, const cv::Point3d& the_point,
                         bool is_first =true, const double epsilon = 1)
//---------------------------------------------------------------------------------------------------------------------------
{
   const double x = the_point.x;
   const double y = the_point.y;
   size_t i = 0;
   for (const std::pair<cv::Point3d, cv::Point3d>& pp : points)
   {
      const cv::Point3d& pt = (is_first) ? pp.first : pp.second;
      const double xi = pt.x;
      const double yi = pt.y;
      if ( (mut::near_zero(x - xi, epsilon)) && (mut::near_zero(y - yi, epsilon)) )
         return i;
      i++;
   }
   return std::numeric_limits<size_t>::max();
}


ImageWindow::ImageWindow(QApplication *app, QWidget *parent) : QMainWindow(parent), panel_layout(new QVBoxLayout(this)),
                                                             app(app), train_image_holder(new CVQtScrollableImage(this)),
                                                             query_image_holder(new CVQtScrollableImage(this)),
                                                             match_image_holder(new CVQtImage(this))
//-----------------------------------------------------------------------------------------------------------------
{
   resize(QGuiApplication::primaryScreen()->availableSize() * 4 / 5);

   QWidget* panel = new QWidget();
   setup_features_control(panel_layout);
   panel->setLayout(panel_layout);
   panel->updateGeometry();
   tabs.addTab(panel, "Feature Controls");
   tabs.addTab(train_image_holder, "Train Image");
   tabs.addTab(query_image_holder, "Query Image");
   tabs.addTab(match_image_holder, "Matches");
   tabs.addTab(pose_layout(), "Pose");
   setCentralWidget(&tabs);
   connect(&tabs, SIGNAL(currentChanged(int)), this, SLOT(tab_selected()));

   std::function<void(cv::Mat&, cv::Rect&, bool, bool, bool)> f = std::bind(&ImageWindow::on_region_selected, this,
                                                                            std::placeholders::_1, std::placeholders::_2,
                                                                            std::placeholders::_3, std::placeholders::_4,
                                                                            std::placeholders::_5);

   train_image_holder->set_roi_callback(f);

   std::function<void(cv::Point2f&, bool, bool, bool)> f2 = std::bind(&ImageWindow::on_left_click, this,
                                                                      std::placeholders::_1, std::placeholders::_2,
                                                                      std::placeholders::_3, std::placeholders::_4);
   train_image_holder->set_left_click_callback(f2);

   std::function<void(cv::Point2f&, bool, bool, bool)> f3 = std::bind(&ImageWindow::on_right_click, this,
                                                                      std::placeholders::_1, std::placeholders::_2,
                                                                      std::placeholders::_3, std::placeholders::_4);
   train_image_holder->set_right_click_callback(f3);

   open_train_img_action = new QAction(tr("&Open Training Image"), this);
   open_train_img_action->setShortcut(QKeySequence(tr("Ctrl+Shift+t")));
   open_train_img_action->setStatusTip("Open an training image");
   connect(open_train_img_action, SIGNAL(triggered(bool)), this, SLOT(open_train_image()));

   open_query_img_action = new QAction(tr("&Open Query Image"), this);
   open_query_img_action->setShortcut(QKeySequence(tr("Ctrl+Shift+q")));
   open_query_img_action->setStatusTip("Open a query image");
   connect(open_query_img_action, SIGNAL(triggered(bool)), this, SLOT(open_query_image()));

   open_calibration_action = new QAction(tr("&Open Calibration File"), this);
   open_calibration_action->setShortcut(QKeySequence(tr("Ctrl+Shift+c")));
   open_calibration_action->setStatusTip("Open a calibration file");
   connect(open_calibration_action, SIGNAL(triggered(bool)), this, SLOT(open_calibration_file()));

   save_action = new QAction(tr("&Save"), this);
   save_action->setShortcut(QKeySequence::Save);
   save_action->setStatusTip("Save Training details");
   connect(save_action, SIGNAL(triggered(bool)), this, SLOT(save()));

   load_action = new QAction(tr("&Load"), this);
   load_action->setShortcut(QKeySequence::Open);
   load_action->setStatusTip("Load Training details");
   connect(load_action, SIGNAL(triggered(bool)), this, SLOT(load()));

   save_matches_action = new QAction(tr("Save Matches"), this);
   save_matches_action->setShortcut(QKeySequence(tr("Ctrl+Shift+s")));
   save_matches_action->setStatusTip("Save matches");
   connect(save_matches_action, SIGNAL(triggered(bool)), this, SLOT(save_matches()));

   load_matches_action = new QAction(tr("&Load Matches"), this);
   load_matches_action->setShortcut(QKeySequence(tr("Ctrl+l")));
   load_matches_action->setStatusTip("Load Matches");
   connect(load_matches_action, SIGNAL(triggered(bool)), this, SLOT(load_matches()));

   load_3dmatches_action = new QAction(tr("&Load 3D Matches"), this);
   load_3dmatches_action->setStatusTip("Load 3D Matches");
   connect(load_3dmatches_action, SIGNAL(triggered(bool)), this, SLOT(load_3dmatches()));

   exit_action = new QAction("E&xit", this);
   connect(exit_action, SIGNAL(triggered(bool)), this, SLOT(exit()));

   detect_action = new QAction("&Detect (uses current settings)", this);
   detect_action->setShortcut(QKeySequence(tr("Ctrl+d")));
   detect_action->setStatusTip("Detect using current settings");
   connect(detect_action, SIGNAL(triggered(bool)), this, SLOT(on_detect()));

   match_action = new QAction("&Match (uses current settings)", this);
   match_action->setShortcut(QKeySequence(tr("Ctrl+m")));
   match_action->setStatusTip("Match using current settings");
   connect(match_action, SIGNAL(triggered(bool)), this, SLOT(on_match()));

   clear_selected_action = new QAction("Clear Selection", this);
   clear_selected_action->setShortcut(QKeySequence(tr("Ctrl+Shift+c")));
   clear_selected_action->setStatusTip("Clear selected keypoints");
   connect(clear_selected_action, SIGNAL(triggered(bool)), this, SLOT(on_clear_selection()));

   selected_from_matched_action = new QAction("Set Selected Keypoints from Matched", this);
   selected_from_matched_action->setStatusTip("Sets the currently selected keypoints from the currently matched ones");
   connect(selected_from_matched_action, SIGNAL(triggered(bool)), this, SLOT(on_selected_from_matched()));

   selected_from_homography_action = new QAction("Set Selected Keypoints from Homography", this);
   selected_from_homography_action->setStatusTip("Sets the currently selected keypoints from the current homography matched ones");
   connect(selected_from_homography_action, SIGNAL(triggered(bool)), this, SLOT(on_selected_from_homography()));

   delete_unselected_action = new QAction("Delete Unselected", this);
   delete_unselected_action->setStatusTip("Delete all unselected keypoints");
   connect(delete_unselected_action, SIGNAL(triggered(bool)), this, SLOT(on_delete_unselected()));


   file_menu = menuBar()->addMenu("&File");
   file_menu->addAction(open_train_img_action);
   file_menu->addAction(open_query_img_action);
   file_menu->addSeparator();
   file_menu->addAction(open_calibration_action);
   file_menu->addSeparator();
   file_menu->addAction(save_action);
   file_menu->addAction(load_action);
   file_menu->addSeparator();
   file_menu->addAction(save_matches_action);
   file_menu->addAction(load_matches_action);
   file_menu->addAction(load_3dmatches_action);
   file_menu->addSeparator();
   file_menu->addAction(exit_action);

   action_menu = menuBar()->addMenu("&Action");
   action_menu->addAction(detect_action);
   action_menu->addAction(match_action);
   action_menu->addSeparator();
   action_menu->addAction(clear_selected_action);
   action_menu->addSeparator();
   action_menu->addAction(selected_from_matched_action);
   action_menu->addAction(selected_from_homography_action);
   action_menu->addSeparator();
   action_menu->addAction(delete_unselected_action);
}

void ImageWindow::tab_selected()
//-----------------------------
{
   switch (tabs.currentIndex())
   {
      case 4:
         if ( (matched_train_keypoints.empty()) && (homography_train_keypoints.empty()) &&
              (threeD_matches.empty()) && (loaded_threeD_matches.empty()) )
         {
            poseButton->setEnabled(false);
            listPoseMethod->setEnabled(false);

         }
         else
         {
            poseButton->setEnabled(true);
            listPoseMethod->setEnabled(true);
         }
         break;
      default: break;
   }
}

void ImageWindow::open_train_image()
//----------------------------------
{
   QString image_file =  QFileDialog::getOpenFileName(this, tr("Open Training Image"), ".",
                                                    tr("Images (*.png *.jpg *.jpeg)"));
   if (! image_file.isEmpty())
   {
      std::stringstream errs;
      if (!load_train_img(image_file.toStdString(), &errs))
      {
         QMessageBox::warning(this, "Open Error", QString(errs.str().c_str()));
         return;
      }
      else if (! errs.str().empty())
         QMessageBox::warning(this, "Open Warning", QString(errs.str().c_str()));
   }
}

void ImageWindow::open_query_image()
//----------------------------------
{
   QString image_file =  QFileDialog::getOpenFileName(this, tr("Open Query Image"), ".",
                                                      tr("Images (*.png *.jpg *.jpeg)"));
   if (! image_file.isEmpty())
   {
      std::stringstream errs;
      if (!load_query_img(image_file.toStdString(), &errs))
         QMessageBox::warning(this, "Open Error", QString(errs.str().c_str()));
      else if (! errs.str().empty())
         QMessageBox::warning(this, "Open Warning", QString(errs.str().c_str()));
   }
}

void ImageWindow::open_calibration_file()
//----------------------------------
{
   QString calibration_file =  QFileDialog::getOpenFileName(this, tr("Open Calibration File"), ".",
                                                      tr("Calibration Files (*.yaml *.csv)"));
   if (! calibration_file.isEmpty())
   {
      std::stringstream errs;
      std::unique_ptr<Calibration> calibration_ptr;
      calibration_ptr.reset(new Calibration);
      if (calibration_ptr->open(calibration_file.toStdString(), &errs))
      {
         calibration.reset(nullptr);
         calibration = std::move(calibration_ptr);
      }
      else
         QMessageBox::warning(this, "Open Calibration Error", QString(errs.str().c_str()));
   }
}

void ImageWindow::save()
//----------------------
{
   if (all_train_keypoints.size() == 0)
   {
      QMessageBox::warning(this, "Save Error", "No keypoints detected (or detect not clicked yet). Cannot save");
      return;
   }
   std::vector<cv::KeyPoint> selected_keypoints;
   cv::Mat selected_descriptors;
   cv::Point2f centroid;
   for (size_t i=0; i<all_train_keypoints.size(); i++)
   {
      cv::KeyPoint& kp = all_train_keypoints[i];
      KeyPointExtra_t kpc;
      kpc.class_id = static_cast<unsigned int>(kp.class_id);
      if (kpc.is_selected == 1)
      {
         selected_keypoints.emplace_back(kp);
         selected_descriptors.push_back(all_train_descriptors.row(i));
         centroid.x += kp.pt.x;
         centroid.y += kp.pt.y;
      }
   }
   if (selected_keypoints.empty())
   {
      QMessageBox::StandardButton reply = QMessageBox::question(this, "No Selections",
                                                                "No keypoints are selected. Do you want to save all keypoints ?",
                                                                QMessageBox::Yes | QMessageBox::No);
      if  (reply == QMessageBox::Yes)
      {
         selected_keypoints = all_train_keypoints;
         selected_descriptors = all_train_descriptors;
         for (cv::KeyPoint &kp : selected_keypoints)
         {
            centroid.x += kp.pt.x;
            centroid.y += kp.pt.y;
         }
      }
      else
         return;
   }
   centroid.x /= static_cast<float>(selected_keypoints.size());
   centroid.y /= static_cast<float>(selected_keypoints.size());

   std::vector<cv::Point2f> train_points;
   cv::KeyPoint::convert(selected_keypoints, train_points);
   cv::RotatedRect rbb;
   cv::Rect2f bb;
   try
   {
      rbb = cv::minAreaRect(train_points);
      bb = rbb.boundingRect2f();
   }
   catch (cv::Exception& cverr)
   {
      rbb = cv::RotatedRect();
      bb = cv::Rect2f(0, 0, 0, 0);
      std::cerr << "Error finding bounding box: " << cverr.what() << std::endl;
   }
   std::vector<cv::Point2f> hull_points;
   cv::Point2f hull_centroid;
   std::vector<cv::KeyPoint> hull_keypoints;
   cv::Mat hull_descriptors;
   int chull = get_convex_hull(selected_keypoints, selected_descriptors, hull_points, hull_keypoints, hull_descriptors,
                               hull_centroid, true);
   if (chull >= 4)
      centroid = hull_centroid;

//   cv::Mat debug_img;
//   cv::drawKeypoints(pre_detect_train_image, selected_keypoints, debug_img, cv::Scalar(0, 255, 0));
//   cv::drawKeypoints(pre_detect_train_image, hull_keypoints, debug_img, cv::Scalar(0, 255, 255),
//                     cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
//   std::vector<cv::KeyPoint> v;
//   v.push_back(cv::KeyPoint(centroid, 0, 0, 0, 0, 0));
//   cv::drawKeypoints(pre_detect_train_image, v, debug_img, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
//   cv::imwrite("keypoints-1.png", debug_img);
//   cv::imshow("win", debug_img); cv::waitKey(0);

//   cv::Mat normal, rotated_normal;
//   if (calibration_values && image_meta)
//   {
//      if (! find_normal(calibration_values.get(), image_meta, hull_keypoints, selected_keypoints, normal, rotated_normal))
//      {
//         normal.release();
//         rotated_normal.release();
//      }
//   }
   std::unique_ptr<SaveDialog> dlg(new SaveDialog(this, train_image_meta.depth));
   if ( (dlg) && (dlg->exec() == 1) && (! dlg->path().empty()) )
   {
      filesystem::path save_dir(dlg->path());
      filesystem::path parent_dir = filesystem::absolute(save_dir).parent_path();
//      std::cout << save_dir.string() << " " << parent_dir.string() << std::endl;
      if (!filesystem::is_directory(parent_dir))
         filesystem::create_directories(parent_dir);
      std::string dir = parent_dir.string();
      filesystem::path p = parent_dir / "id.detect";
      std::ofstream ofs(filesystem::absolute(p).string());
      ofs << dlg->id() << "\n" << dlg->rid() << "\n";
      ofs.close();
      ocv_write(pre_detect_train_image_bw, pre_detect_train_image, selected_keypoints, selected_descriptors,
                hull_keypoints, hull_descriptors, centroid,
                feature_detector_name, descriptor_extractor_name, dlg->id(), dlg->rid(), dlg->description(), dir,
                dlg->latitude(), dlg->longitude(), dlg->altitude(), dlg->depth(), bb, rbb, train_image_meta);
      json_write(pre_detect_train_image_bw, pre_detect_train_image, selected_keypoints, selected_descriptors,
                 hull_keypoints, hull_descriptors, centroid,
                 feature_detector_name, descriptor_extractor_name, dlg->id(), dlg->rid(), dlg->description(), dir,
                 dlg->latitude(), dlg->longitude(), dlg->altitude(), dlg->depth(), bb, rbb, train_image_meta);
      yaml_write(pre_detect_train_image_bw, pre_detect_train_image, selected_keypoints, selected_descriptors,
                 hull_keypoints, hull_descriptors, centroid,
                 feature_detector_name, descriptor_extractor_name, dlg->id(), dlg->rid(), dlg->description(), dir,
                 dlg->latitude(), dlg->longitude(), dlg->altitude(), dlg->depth(), bb, rbb, train_image_meta);
      if ( (! matched_train_keypoints.empty()) || (! matched_query_keypoints.empty()) ||
           (! homography_train_keypoints.empty()) || (! homography_query_keypoints.empty()) )
      {
         std::stringstream ss;
         ss << dlg->id() << "-" << dlg->rid();
         std::string name = ss.str();
         yaml_write_matches(dir, name, matched_train_keypoints, matched_train_descriptors,
                            matched_query_keypoints, matched_query_descriptors,
                            homography_train_keypoints, homography_train_descriptors,
                            homography_query_keypoints, homography_query_descriptors,
                            matches, homography_matches, matched_points, homography_matched_points);
         if (! threeD_matches.empty())
            yaml_write_3Dmatches(dir, name, threeD_matches);
         ocv_write_matches(dir, name, matched_train_keypoints, matched_train_descriptors,
                           matched_query_keypoints, matched_query_descriptors,
                           homography_train_keypoints, homography_train_descriptors,
                           homography_query_keypoints, homography_query_descriptors,
                           matches, homography_matches);
         json_write_matches(dir, name, matched_train_keypoints, matched_train_descriptors,
                            matched_query_keypoints, matched_query_descriptors,
                            homography_train_keypoints, homography_train_descriptors,
                            homography_query_keypoints, homography_query_descriptors,
                            matches, homography_matches);
      }
      if (train_image_meta)
      {
         std::string ext = train_image_meta.metafile.extension();
         filesystem::path metafile = parent_dir / (dlg->rids() + "-meta" + ext);
         try
         { filesystem::copy(train_image_meta.metafile, metafile); } catch (std::exception& e) { std::cerr << "Error copying " << train_image_meta.metafile << " on save (" << e.what() << ")\n"; }
      }
      std::string queryname = dlg->rids() + "-query";
      if (query_image_meta)
      {
         std::string ext = train_image_meta.metafile.extension();
         filesystem::path metafile = parent_dir / (queryname + ext);
         try
         { filesystem::copy(query_image_meta.metafile, metafile); } catch (std::exception& e) { std::cerr << "Error copying " << query_image_meta.metafile << " on save (" << e.what() << ")\n"; }
      }
      p = parent_dir / (queryname + ".png");
      cv::imwrite(p.c_str(), pre_detect_query_image);
      p = parent_dir / (queryname + "-bw.png");
      cv::imwrite(p.c_str(), pre_detect_query_image_bw);

      if ( (calibration) || (train_image_meta.calibration) )
      {
         p = parent_dir / "calibration.yaml";
         if (calibration)
            calibration->save(p.string(), true);
         else
            train_image_meta.calibration->save(p.string(), true);
      }
   }
}

void ImageWindow::save_matches()
//------------------------------
{
   if ( (matched_train_keypoints.empty()) || (matched_query_keypoints.empty()) )
   {
      QMessageBox::warning(this, "Save Error", "No matched keypoints detected (or match not clicked yet). Cannot save");
      return;
   }
   std::unique_ptr<SaveMatchDialog> dlg(new SaveMatchDialog(this, train_image_meta.depth));
   if ( (dlg) && (dlg->exec() == 1) && (! dlg->path().empty()) )
   {
      filesystem::path save_dir(dlg->path());
      save_dir = filesystem::absolute(save_dir);
      filesystem::create_directories(save_dir);
      std::string name = trim(dlg->name());
      filesystem::path p = save_dir / (name + ".match");
      std::ofstream mos(p.string());
      mos.close();
      p = save_dir / (name + "-train");
      std::string train_img_filepath = p.string();
      cv::imwrite((train_img_filepath + ".png").c_str(), pre_detect_train_image);
      cv::imwrite((train_img_filepath + "-bw.png").c_str(), pre_detect_train_image);
      p = save_dir / (name + "-query");
      std::string query_img_filepath = p.string();
      cv::imwrite((query_img_filepath + ".png").c_str(), pre_detect_query_image);
      cv::imwrite((query_img_filepath + "-bw.png").c_str(), pre_detect_query_image_bw);
      if ( (train_image_meta) && (filesystem::is_regular_file(train_image_meta.metafile)) )
      {
         std::string ext = train_image_meta.metafile.extension();
         try { filesystem::copy(train_image_meta.metafile, train_img_filepath + ext); } catch (std::exception& e) { std::cerr << "Error copying " << train_image_meta.metafile << " on save " << e.what() << std::endl; }
      }
      if ( (query_image_meta) && (filesystem::is_regular_file(query_image_meta.metafile)) )
      {
         std::string ext = query_image_meta.metafile.extension();
         try { filesystem::copy(query_image_meta.metafile, query_img_filepath + ext); } catch (std::exception& e) { std::cerr << "Error copying " << query_image_meta.metafile << " on save " << e.what() << std::endl; }
      }
      bool write_error = false;
      std::stringstream errs;
      if (! yaml_write_matches(save_dir.string(), name, matched_train_keypoints, matched_train_descriptors, matched_query_keypoints, matched_query_descriptors,
                               homography_train_keypoints, homography_train_descriptors, homography_query_keypoints, homography_query_descriptors,
                               matches, homography_matches, matched_points, homography_matched_points, &errs))
         write_error = true;
      if (! threeD_matches.empty())
         yaml_write_3Dmatches(save_dir.string(), name, threeD_matches);
      json_write_matches(save_dir.string(), name, matched_train_keypoints, matched_train_descriptors, matched_query_keypoints, matched_query_descriptors,
                         homography_train_keypoints, homography_train_descriptors, homography_query_keypoints, homography_query_descriptors,
                         matches, homography_matches);
      ocv_write_matches(save_dir.string(), name, matched_train_keypoints, matched_train_descriptors, matched_query_keypoints, matched_query_descriptors,
                        homography_train_keypoints, homography_train_descriptors, homography_query_keypoints, homography_query_descriptors,
                        matches, homography_matches);
      if ( (calibration) || (train_image_meta.calibration) )
      {
         p = save_dir / (name + "-calibration.yaml");
         if (calibration)
            calibration->save(p.string(), true);
         else
            train_image_meta.calibration->save(p.string(), true);
      }
      if (write_error)
      {
         QMessageBox::warning(this, "Save Error", QString("Error saving matches ") + QString(errs.str().c_str()));
//         filesystem::remove_all(save_dir);
      }
   }
}

inline cv::KeyPoint YAML_keypoint(const YAML::Node& kpn)
//------------------------------------------------------
{
   if (! kpn) return cv::KeyPoint(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(), 0, 0, 0, 0, 0);
   float x = (kpn["x"]) ? kpn["x"].as<float>() : std::numeric_limits<float>::quiet_NaN();
   float y = (kpn["y"]) ? kpn["y"].as<float>() : std::numeric_limits<float>::quiet_NaN();
   float angle = (kpn["angle"]) ? kpn["angle"].as<float>() : std::numeric_limits<float>::quiet_NaN();
   float response = (kpn["response"]) ? kpn["response"].as<float>() : std::numeric_limits<float>::quiet_NaN();
   float size = (kpn["size"]) ? kpn["size"].as<float>() : std::numeric_limits<float>::quiet_NaN();
   int octave = (kpn["octave"]) ? kpn["octave"].as<float>() : 0;
   return cv::KeyPoint(x, y, size, angle, response, octave, 0);
}

inline cv::Point3d YAML_point3d(const YAML::Node& kpn)
//------------------------------------------------------
{
   if (! kpn) return cv::Point3d(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN());
   float x = (kpn["x"]) ? kpn["x"].as<float>() : std::numeric_limits<float>::quiet_NaN();
   float y = (kpn["y"]) ? kpn["y"].as<float>() : std::numeric_limits<float>::quiet_NaN();
   float z = (kpn["z"]) ? kpn["z"].as<float>() : std::numeric_limits<float>::quiet_NaN();
   return cv::Point3d(x, y, z);
}

inline void load_matches_keypoints(YAML::Node& n, std::vector<cv::KeyPoint>* all, std::vector<cv::KeyPoint>* matched, std::vector<cv::KeyPoint>* homography)
//-------------------------------------------------------------------------------------------------------------------------------------------
{
   for (auto it = n.begin(); it != n.end(); ++it)
   {
      YAML::Node kpn = *it;
      if (! kpn) continue;
      cv::KeyPoint kp = YAML_keypoint(kpn);
      if (all != nullptr)
         all->emplace_back(kp);
      if (matched != nullptr)
         matched->emplace_back(kp);
      if (homography != nullptr)
         homography->emplace_back(kp);
   }
}
template <typename T>
void YAML_load_mat(const YAML::Node& N, cv::Mat& m, int type=-1)
//---------------------------------------------------------
{
   if (! N) m = cv::Mat::zeros(1, 1, type);
   const YAML::Node& data = N["data"];
   if (! data) throw std::logic_error("ImageWindow::YAML_load_mat:  Data node not found.");
   size_t i = 0;
   int rows = (N["rows"]) ? N["rows"].as<int>() : 0;
   int cols = (N["cols"]) ? N["cols"].as<int>() : 0;
   if (type != -1)
      m.create(rows, cols, type);
   for (int row = 0; row < rows; row++)
   {
      for (int col = 0; ( (col < cols) && (i < data.size()) ); col++)
      {
         YAML::Node n = data[i++];
         if (n)
            m.at<T>(row, col) = n.as<T>();
      }
   }
}

void YAML_load_matches(const YAML::Node& n, std::vector<cv::DMatch>& matches)
//---------------------------------------------------------------------------
{
   for (auto it = n.begin(); it != n.end(); ++it)
   {
      YAML::Node mn = *it;
      if (! mn) continue;
      int trainIdx = (mn["trainIdx"]) ? mn["trainIdx"].as<int>() : -1;
      int queryIdx = (mn["queryIdx"]) ? mn["queryIdx"].as<int>() : -1;
      float distance = (mn["distance"]) ? mn["distance"].as<float>() : std::numeric_limits<float>::quiet_NaN();
      int imgIdx = (mn["imgIdx"]) ? mn["imgIdx"].as<int>() : -1;
      matches.emplace_back(queryIdx, trainIdx, imgIdx, distance);
   }
}

inline void YAML_load_matched_points(const YAML::Node& n, std::vector<std::pair<cv::Point3d, cv::Point3d>>& matched_pts)
//----------------------------------------------------------------------------------------------------------------------
{
   if (! n) return;
   for (auto it = n.begin(); it != n.end(); ++it)
   {
      YAML::Node mn = *it;
      if (! mn) continue;
      YAML::Node first = mn["first"], second = mn["second"];
      if ( (first) && (second) )
      {
         cv::Point3d one(first["x"].as<double>(), first["y"].as<double>(), first["z"].as<double>());
         cv::Point3d two(second["x"].as<double>(), second["y"].as<double>(), second["z"].as<double>());
         matched_pts.emplace_back(std::make_pair(one, two));
      }
   }
}

void YAML_load_3D_matches(const YAML::Node& n, std::vector<std::tuple<cv::Point3d, cv::KeyPoint, cv::Mat>>& threeD_matches)
//-------------------------------------------------------------------------------------------------------------------------
{
   for (auto it = n.begin(); it != n.end(); ++it)
   {
      YAML::Node mn = *it;
      if (!mn) continue;
      cv::Point3d pt = YAML_point3d(mn["Pt"]);
      cv::KeyPoint keypoint = YAML_keypoint(mn["Kp"]);
      cv::Mat descriptor;
      YAML_load_mat<float>(mn["Descriptor"], descriptor, CV_32FC1);
      threeD_matches.emplace_back(std::make_tuple(pt, keypoint, descriptor));
   }
}

inline void load_matches_descriptors(YAML::Node& n, cv::Mat* all, cv::Mat* matched, cv::Mat* homography)
//-------------------------------------------------------------------------------------------------------
{
   if (all != nullptr)
      YAML_load_mat<float>(n, *all, CV_32FC1);
   if (matched != nullptr)
      YAML_load_mat<float>(n, *matched, CV_32FC1);
   if (homography != nullptr)
      YAML_load_mat<float>(n, *homography, CV_32FC1);
}

void ImageWindow::load()
//----------------------
{
   filesystem::path cwd(filesystem::current_path());
   QString mfile = QFileDialog::getOpenFileName(this, tr("Open"), filesystem::absolute(cwd).string().c_str(),
                                                tr("Detect (*.detect)"));
   if (mfile.isEmpty()) return;
   filesystem::path idfile(mfile.toStdString());
   if (! filesystem::exists(idfile)) return;
   idfile = filesystem::absolute(idfile);
   filesystem::path directory = filesystem::absolute(idfile.parent_path());
   if (! filesystem::is_directory(directory))
   {
      QMessageBox::warning(this, "Open", QString(directory.c_str()) + " not a valid directory.");
      return;
   }
   std::string dir = directory.string();
   std::ifstream ifs(idfile);
   std::string id, rid;
   ifs >> id >> rid;
   std::string train_imgname = rid;

   std::stringstream errs;
   cv::Mat emptyimg;
   train_image_holder->set_image(emptyimg);
   pre_detect_train_image.release();
   if (! load_image(dir + "/" + rid + "-color.png", pre_detect_train_image, pre_detect_train_image_bw,
                    train_image_holder, train_image_meta, &errs, dir + "/" + rid + "-meta.yaml"))
   {
      QMessageBox::warning(this, "Open Match",  QString("Error loading training image file ") + QString(errs.str().c_str()));
      return;
   }

   query_image_holder->set_image(emptyimg);
   pre_detect_query_image.release();
   filesystem::path p = directory / (rid + "-query.png");
   if (filesystem::exists(p))
      load_image(p.string(), pre_detect_query_image, pre_detect_query_image_bw,
                 query_image_holder, query_image_meta);

   p = directory / (rid + ".yaml");
   if (filesystem::exists(p))
   {
      YAML::Node root = YAML::LoadFile(p.string());
      if (train_image_meta.deviceGravity.empty())
      {
         YAML::Node n = root["device_gravity"];
         if (n)
            YAML_load_mat<float>(n, train_image_meta.deviceGravity, CV_64FC1);
      }
      if (train_image_meta.correctedGravity.empty())
      {
         YAML::Node n = root["gravity"];
         if (n)
            YAML_load_mat<float>(n, train_image_meta.correctedGravity, CV_64FC1);
      }
      if ((std::isnan(train_image_meta.depth)) || (train_image_meta.depth <= 0))
      {
         YAML::Node n = root["depth"];
         if (n)
            train_image_meta.depth = n.as<float>();
      }
      all_train_keypoints.clear();
      all_query_keypoints.clear();
      if (!all_train_descriptors.empty())
         all_train_descriptors.release();
      if (!all_query_descriptors.empty())
         all_query_descriptors.release();
      matched_train_keypoints.clear();
      matched_query_keypoints.clear();
      if (!matched_train_descriptors.empty())
         matched_train_descriptors.release();
      if (!matched_query_descriptors.empty())
         matched_query_descriptors.release();
      homography_train_keypoints.clear();
      homography_query_keypoints.clear();
      if (!homography_train_descriptors.empty())
         homography_train_descriptors.release();
      if (!homography_query_descriptors.empty())
         homography_query_descriptors.release();
      matches.clear();
      homography_matches.clear();
      threeD_matches.clear();
      YAML::Node n = root["keypoints"];
      if (n)
         load_matches_keypoints(n, &all_train_keypoints, nullptr, nullptr);
      n = root["descriptors"];
      if (n)
         YAML_load_mat<float>(n, all_train_descriptors, CV_32FC1);

      cv::Mat img;
      drawKeypoints(pre_detect_train_image, all_train_keypoints, img);
      train_image_holder->set_image(img);
      index_points.set(&all_train_keypoints);
      index.reset(new keypoint_kd_tree_t(2, index_points, nanoflann::KDTreeSingleIndexAdaptorParams(10)));
      index->buildIndex();

      filesystem::path matchfile = directory / (id + "-" + rid + "-matches.yaml");
      if (filesystem::exists(matchfile))
      {
         YAML::Node root = YAML::LoadFile(matchfile.string());
         YAML::Node n = root["matched_train_keypoints"];
         if (n)
            load_matches_keypoints(n, nullptr, &matched_train_keypoints, nullptr);
         n = root["matched_train_descriptors"];
         if (n)
            load_matches_descriptors(n, nullptr, &matched_train_descriptors, nullptr);
         n = root["matched_query_keypoints"];
         if (n)
            load_matches_keypoints(n, nullptr, &matched_query_keypoints, nullptr);
         n = root["matched_query_descriptors"];
         if (n)
            load_matches_descriptors(n, nullptr, &matched_query_descriptors, nullptr);
         n = root["homography_train_keypoints"];
         if (n)
            load_matches_keypoints(n, nullptr, nullptr, &homography_train_keypoints);
         n = root["homography_train_descriptors"];
         if (n)
            load_matches_descriptors(n, nullptr, nullptr, &homography_train_descriptors);
         n = root["homography_query_keypoints"];
         if (n)
            load_matches_keypoints(n, nullptr, nullptr, &homography_query_keypoints);
         n = root["homography_query_descriptors"];
         if (n)
            load_matches_descriptors(n, nullptr, nullptr, &homography_query_descriptors);
         n = root["matches"];
         YAML_load_matches(n, matches);
         n = root["homography_matches"];
         YAML_load_matches(n, homography_matches);
         n = root["matched_points"];
         YAML_load_matched_points(n, matched_points);
         n = root["homography_matched_points"];
         YAML_load_matched_points(n, homography_matched_points);
         if (!pre_detect_query_image.empty())
         {
            cv::Mat img;
            drawKeypoints(pre_detect_query_image, matched_query_keypoints, img);
            query_image_holder->set_image(img);
            if (!homography_train_keypoints.empty())
            {
               cv::Mat show_matches;
               std::vector<cv::Point2f> train_convex_hull_pts, query_convex_hull_pts;
               cv::Point2f centroid;
               cv::RotatedRect train_rbb, query_rbb;
               cv::Rect2f train_bb, query_bb;
               errs.str("");
               if ((ocv::convex_hull(homography_train_keypoints, train_convex_hull_pts, centroid, train_rbb, train_bb,
                                     nullptr, &errs)) &&
                   (ocv::convex_hull(homography_query_keypoints, query_convex_hull_pts, centroid, query_rbb, query_bb,
                                     nullptr, &errs)))
               {
                  cv::Mat train_image, query_image;
                  pre_detect_train_image.copyTo(train_image);
                  cv::rectangle(train_image, train_bb, red, 2);
                  pre_detect_query_image.copyTo(query_image);
                  cv::rectangle(query_image, query_bb, red, 2);
                  match_image_holder->set_image(ocv::draw_matches(train_image, matched_train_keypoints,
                                                                  query_image, matched_query_keypoints,
                                                                  matches, show_matches));
               }
            }
         }
      }
      matchfile = directory / (id + "-" + rid + "-matches3d.yaml");
      if (filesystem::exists(matchfile))
      {
         YAML::Node root = YAML::LoadFile(matchfile.string());
         n = root["3d_matches"];
         if (n)
         {
            YAML_load_3D_matches(n, threeD_matches);
            index_points3d.set(&threeD_matches);
            index3d.reset(new threeD_kd_tree_t(2, index_points3d, nanoflann::KDTreeSingleIndexAdaptorParams(10)));
            index3d->buildIndex();
         }
      }
   }
   p = directory / "calibration.yaml";
   if (filesystem::exists(p))
   {
      errs.str("");
      calibration.reset(new Calibration);
      if (! calibration->open(p.string(), &errs))
      {
         errs << "Error loading calibration file " << p.string() << " (" << errs.str() << ")";
         calibration.reset(nullptr);
         msg(errs);
      }
   }
}

inline double double_from_string(std::string s) { try { return std::stod(trim(s)); } catch (...) { return mut::NaN; }}
inline double float_from_string(std::string s) { try { return std::stof(trim(s)); } catch (...) { return mut::NaN_F; }}

//Format x,y,z -> u,v eg 0,0,0 -> 1082,549
//TODO: Allow right click on random point in query image to open dialog allowing entry of corresponding 3D point in train
// image, and saving in the above format.
void ImageWindow::load_3dmatches()
//-------------------------------
{
   if ( (pre_detect_train_image.empty()) || (pre_detect_query_image.empty()) )
   {
      msg("First load relevant training and query images");
      return;
   }
   filesystem::path cwd(filesystem::current_path());
   QString mfile = QFileDialog::getOpenFileName(this, tr("Open Match"), filesystem::absolute(cwd).string().c_str(),
                                                tr("Matches (*.3d)"));
   if (mfile.isEmpty()) return;
   filesystem::path matchfile(mfile.toStdString());
   if (! filesystem::exists(matchfile)) return;
   matchfile = filesystem::absolute(matchfile);
   std::string matchname = matchfile.stem();
   std::ifstream ifs(matchfile);
   if (! ifs)
   {
      std::cerr << "Could not open match file " << matchfile << " (" << std::strerror(errno)
                << ")" << std::endl;
      std::exit(1);
   }
   char buf[120];
   ifs.getline(buf, 120);
   int c = 1, ypos = -1;
   cv::Mat img;
   pre_detect_query_image.copyTo(img);
   loaded_threeD_matches.clear();
   while (!ifs.eof())
   {
      std::string line = trim(buf);
      if ( (line.empty()) || (line[0] == '#') )
      {
         ifs.getline(buf, 120);
         continue;
      }
      auto pos = line.find('#');
      if (pos != std::string::npos)
         line = line.substr(0, pos);
      std::string target;
      pos = line.find("->");
      if (pos == std::string::npos)
      {
         pos = line.find('=');
         if (pos == std::string::npos)
         {
            std::cerr << "Invalid line " << line << " in match file " << matchfile << std::endl;
            std::exit(1);
         }
         else
            try { target = trim(line.substr(pos + 1)); } catch (...) { target = ""; }
      }
      else
         try { target = trim(line.substr(pos + 2)); } catch (...) { target = ""; }
      std::string source = trim(line.substr(0, pos));
      std::vector<std::string> tokens;
      size_t n = split(source, tokens, ",");
      if ( (n < 2) || (n > 3) )
      {
         std::cerr << "Invalid line " << line << " in match file " << matchfile << std::endl;
         std::exit(1);
      }
      double tx = double_from_string(tokens[0]);
      double ty = double_from_string(tokens[1]);
      double tz;
      if (n > 2)
         tz = double_from_string(tokens[2]);
      else
         tz = 1;
      if ( (std::isnan(tx)) || (std::isnan(ty)) || (std::isnan(tz)) )
      {
         std::cerr << "Invalid line " << line << " in match file " << matchfile << std::endl;
         std::exit(1);
      }

      n = split(target, tokens, ",");
      if (n != 2)
      {
         std::cerr << "Invalid line " << line << " in match file " << matchfile << std::endl;
         std::exit(1);
      }
      double qx = double_from_string(tokens[0]);
      double qy = double_from_string(tokens[1]);
      if ( (std::isnan(qx)) || (std::isnan(qy)) )
      {
         std::cerr << "Invalid line " << line << " in match file " << matchfile << std::endl;
         std::exit(1);
      }
      int xp = cvRound(qx - 1);
      int yp = cvRound(qy - 6);
      std::stringstream ss;
      ss << " (" << tx << ", " << ty << ", " << tz << ") -> (" << qx << ", " << qy << ")";
         if ( (ypos >= 0) && (std::abs(ypos-yp) < 5) )
         {
            yp -= 15;
            xp -= 5;
         }
         ypos = yp;

      ocv::plot_circles(img, tx, ty);
      ocv::cvLabel(img, ss.str(), cv::Point2i(std::max(xp, 0), std::max(yp, 0)));
      ocv::plot_circles(img, qx, qy);

      loaded_threeD_matches.emplace_back(std::make_pair(cv::Point3d(tx, ty, tz), cv::Point2d(qx, qy)));

      ifs.getline(buf, 120);
   }
   if (! loaded_threeD_matches.empty())
      query_image_holder->set_image(img);
}

void ImageWindow::load_matches()
//----------------------
{
   filesystem::path cwd(filesystem::current_path());
   QString mfile = QFileDialog::getOpenFileName(this, tr("Open Match"), filesystem::absolute(cwd).string().c_str(),
                                                tr("Matches (*.match)"));
   if (mfile.isEmpty()) return;
   filesystem::path matchfile(mfile.toStdString());
   if (! filesystem::exists(matchfile)) return;
   matchfile = filesystem::absolute(matchfile);
   std::string matchname = matchfile.stem();
   filesystem::path directory = matchfile.parent_path();
   if (! filesystem::is_directory(directory))
   {
      QMessageBox::warning(this, "Open Match", QString(directory.c_str()) + " not a valid directory.");
      return;
   }
   matchfile = directory / (matchname + "-matches.yaml");
   if (! filesystem::exists(matchfile))
   {
      QMessageBox::warning(this, "Open Match", QString(matchfile.c_str()) + " not found in " + QString(directory.c_str()));
      return;
   }
   filesystem::path train_imgfile = directory / (matchname + "-train.png");
   if (! filesystem::exists(train_imgfile))
   {
      QMessageBox::warning(this, "Open Match", QString(train_imgfile.c_str()) + " not found in " + QString(directory.c_str()));
      return;
   }
   std::stringstream errs;
   if (! load_image(train_imgfile.string(), pre_detect_train_image, pre_detect_train_image_bw,
                    train_image_holder, train_image_meta, &errs))
   {
      QMessageBox::warning(this, "Open Match",  QString(train_imgfile.c_str()) + " error  " + QString(errs.str().c_str()));
      return;
   }
   filesystem::path query_imgfile = directory / (matchname + "-query.png");
   load_image(query_imgfile.string(), pre_detect_query_image, pre_detect_query_image_bw,
              query_image_holder, query_image_meta, nullptr);

   YAML::Node root = YAML::LoadFile(matchfile.string());
   all_train_keypoints.clear(); all_query_keypoints.clear();
   if (! all_train_descriptors.empty())
      all_train_descriptors.release();
   if (! all_query_descriptors.empty())
      all_query_descriptors.release();
   matched_train_keypoints.clear();
   matched_query_keypoints.clear();
   if (! matched_train_descriptors.empty())
      matched_train_descriptors.release();
   if (! matched_query_descriptors.empty())
      matched_query_descriptors.release();
   homography_train_keypoints.clear();
   homography_query_keypoints.clear();
   if (! homography_train_descriptors.empty())
      homography_train_descriptors.release();
   if (! homography_query_descriptors.empty())
      homography_query_descriptors.release();
   matches.clear();
   homography_matches.clear();
   threeD_matches.clear();
   YAML::Node n = root["matched_train_keypoints"];
   if (n)
      load_matches_keypoints(n, &all_train_keypoints, &matched_train_keypoints, nullptr);
   n = root["matched_train_descriptors"];
   if (n)
      load_matches_descriptors(n, &all_train_descriptors, &matched_train_descriptors, nullptr);
   n = root["matched_query_keypoints"];
   if (n)
      load_matches_keypoints(n, &all_query_keypoints, &matched_query_keypoints, nullptr);
   n = root["matched_query_descriptors"];
   if (n)
      load_matches_descriptors(n, &all_query_descriptors, &matched_query_descriptors, nullptr);
   n = root["homography_train_keypoints"];
   if (n)
      load_matches_keypoints(n, nullptr, nullptr, &homography_train_keypoints);
   n = root["homography_train_descriptors"];
   if (n)
      load_matches_descriptors(n, nullptr, nullptr, &homography_train_descriptors);
   n = root["homography_query_keypoints"];
   if (n)
      load_matches_keypoints(n, nullptr, nullptr, &homography_query_keypoints);
   n = root["homography_query_descriptors"];
   if (n)
      load_matches_descriptors(n, nullptr, nullptr, &homography_query_descriptors);
   n = root["matches"];
   YAML_load_matches(n, matches);
   n = root["homography_matches"];
   YAML_load_matches(n, homography_matches);
   n = root["matched_points"];
   YAML_load_matched_points(n, matched_points);
   n = root["homography_matched_points"];
   YAML_load_matched_points(n, homography_matched_points);

   matchfile = directory / (matchname + "-matches3d.yaml");
   if (filesystem::exists(matchfile))
   {
      YAML::Node root = YAML::LoadFile(matchfile.string());
      n = root["3d_matches"];
      if (n)
      {
         YAML_load_3D_matches(n, threeD_matches);
         index_points3d.set(&threeD_matches);
         index3d.reset(new threeD_kd_tree_t(2, index_points3d, nanoflann::KDTreeSingleIndexAdaptorParams(10)));
         index3d->buildIndex();
      }
   }

   filesystem::path p = directory / (matchname + "-calibration.yaml");
   if (filesystem::exists(p))
   {
      errs.str("");
      calibration.reset(new Calibration);
      if (! calibration->open(p.string(), &errs))
      {
         calibration.reset(nullptr);
         errs << "Error loading calibration file " << p.string() << " (" << errs.str() << ")";
         msg(errs);
      }
   }

   cv::Mat img;
   drawKeypoints(pre_detect_train_image, all_train_keypoints, img);
   train_image_holder->set_image(img);
   drawKeypoints(pre_detect_query_image, all_query_keypoints, img);
   query_image_holder->set_image(img);

   index_points.set(&all_train_keypoints);
   index.reset(new keypoint_kd_tree_t(2, index_points, nanoflann::KDTreeSingleIndexAdaptorParams(10)));
   index->buildIndex();

   cv::Mat show_matches;
   std::vector<cv::Point2f> train_convex_hull_pts, query_convex_hull_pts;
   cv::Point2f centroid;
   cv::RotatedRect train_rbb, query_rbb;
   cv::Rect2f train_bb, query_bb;
   errs.str("");
   if ( (ocv::convex_hull(homography_train_keypoints, train_convex_hull_pts, centroid, train_rbb, train_bb, nullptr, &errs)) &&
        (ocv::convex_hull(homography_query_keypoints, query_convex_hull_pts, centroid, query_rbb, query_bb, nullptr, &errs)) )
   {
      cv::Mat train_image, query_image;
      pre_detect_train_image.copyTo(train_image);
      cv::rectangle(train_image, train_bb, red, 2);
      pre_detect_query_image.copyTo(query_image);
      cv::rectangle(query_image, query_bb, red, 2);
      match_image_holder->set_image(ocv::draw_matches(train_image, matched_train_keypoints,
                                                      query_image, matched_query_keypoints,
                                                      matches, show_matches));
   }
}

void ImageWindow::on_left_click(cv::Point2f& pt, bool is_shift, bool is_ctrl, bool is_alt)
//-----------------------------------------------------------------------------------------
{
   cv::KeyPoint* kp;
   float distance;
   if (closest_keypoint(pt.x, pt.y, 10, kp, distance))
   {
      KeyPointExtra_t kpc;
      kpc.class_id = static_cast<unsigned int>(kp->class_id);
      if (kpc.is_selected == 0)
         kpc.is_selected = 1;
      else
         kpc.is_selected = 0;
      kp->class_id = kpc.class_id;
      cv::Mat img;
      drawKeypoints(pre_detect_train_image, all_train_keypoints, img);
      train_image_holder->set_image(img);
   }
}

void ImageWindow::on_right_click(cv::Point2f& pt, bool is_shift, bool is_ctrl, bool is_alt)
//-----------------------------------------------------------------------------------------
{
   cv::KeyPoint* kp;
   float distance;
   if (closest_keypoint(pt.x, pt.y, 15, kp, distance))
   {
      std::function<void(cv::KeyPoint&, cv::Mat&, cv::Point3d&)> f = std::bind(&ImageWindow::on_accept_3d_point, this,
                                                                               std::placeholders::_1, std::placeholders::_2,
                                                                               std::placeholders::_3);
      size_t index = find_keypoint(all_train_keypoints, *kp);
      cv::Mat desc, *pdesc = nullptr;
      if (index < all_train_keypoints.size())
      {
         desc = all_train_descriptors.row(index);
         pdesc = &desc;
      }

      float distance = -1;
      if (closest_3D_point(kp->pt.x, kp->pt.y, 5, edit_3D_tuple, distance))
      {
         const cv::Point3d& p3d = std::get<0>(*edit_3D_tuple);
         enter_3D_coords = new ThreeDCoordsDialog(*kp, pdesc, f, this, p3d.z, p3d.x, p3d.y);
      }
      else
      {
         edit_3D_tuple = nullptr;
         enter_3D_coords = new ThreeDCoordsDialog(*kp, pdesc, f, this, train_image_meta.depth);
      }
      enter_3D_coords->setAttribute(Qt::WA_DeleteOnClose);
      enter_3D_coords->move(pt.x, pt.y);
      enter_3D_coords->show();
      enter_3D_coords->raise();
   }
}

void ImageWindow::on_accept_3d_point(cv::KeyPoint& keypoint, cv::Mat& descriptor, cv::Point3d& pt)
//------------------------------------------------------------------------------------------------
{
   if (edit_3D_tuple == nullptr)
      threeD_matches.emplace_back(std::make_tuple(pt, keypoint, descriptor));
   else
   {
      cv::Point3d& p3d = std::get<0>(*edit_3D_tuple);
      p3d.x = pt.x; p3d.y = pt.y; p3d.z = pt.z;
   }
   index_points3d.set(&threeD_matches);
   index3d.reset(new threeD_kd_tree_t(2, index_points3d, nanoflann::KDTreeSingleIndexAdaptorParams(10)));
   index3d->buildIndex();
}

void ImageWindow::on_region_selected(cv::Mat &roi, cv::Rect& roirect, bool is_shift, bool is_ctrl, bool is_alt)
//--------------------------------------------------------------------------------------------------------------
{
   region_features.clear();
   cv::Mat m(pre_detect_train_image, roirect);
//   cv::imwrite("roi.png", m); cv::imwrite("roi2.png", roi);

   for (size_t i=0; i<all_train_keypoints.size(); i++)
   {
      cv::KeyPoint& kp = all_train_keypoints[i];
      if (roirect.contains(kp.pt))
      {
//         cv::KeyPoint kp2(kp);
//         kp2.pt.x -= roirect.x;
//         kp2.pt.y -= roirect.y;
         cv::Mat descriptor = all_train_descriptors.row(i);

         KeyPointExtra_t kpc;
         kpc.class_id = static_cast<unsigned int>(kp.class_id);
         if (is_ctrl) // Ctrl drag deselects all
            kpc.is_selected = 0;
         else if (kpc.is_selected == 0)
            kpc.is_selected = 1;
         else if (! is_shift) // Shift drag does not toggle already selected points
            kpc.is_selected = 0;
         kp.class_id = kpc.class_id;
//         kp = kp2;

         region_features.emplace_back(i, kp, descriptor, false);
      }
   }
   if (region_features.size() > 0)
   {
      cv::Mat img;
      drawKeypoints(pre_detect_train_image, all_train_keypoints, img);
      train_image_holder->set_image(img);
      tabs.setCurrentIndex(TRAIN_IMAGE_TAB);
   }
}

void ImageWindow::on_selected_color_change()
//-----------------------------------------
{
   QColor color = QColorDialog::getColor(selected_colour, this);
   if( color.isValid() )
   {
      selected_colour = color;
      QString qss = QString("background-color: %1").arg(selected_colour.name());
      selectedColorButton->setStyleSheet(qss);
      cv::Mat img;
      if (! pre_detect_train_image.empty())
      {
         drawKeypoints(pre_detect_train_image, all_train_keypoints, img);
         train_image_holder->set_image(img);
      }
      if (! pre_detect_query_image.empty())
      {
         drawKeypoints(pre_detect_query_image, all_query_keypoints, img);
         query_image_holder->set_image(img);
      }
   }
}

void ImageWindow::on_unselected_color_change()
//-----------------------------------------
{
   QColor color = QColorDialog::getColor(unselected_colour, this);
   if( color.isValid() )
   {
      unselected_colour = color;
      QString qss = QString("background-color: %1").arg(unselected_colour.name());
      unselectedColorButton->setStyleSheet(qss);
      cv::Mat img;
      if (! pre_detect_train_image.empty())
      {
         drawKeypoints(pre_detect_train_image, all_train_keypoints, img);
         train_image_holder->set_image(img);
      }
      if (! pre_detect_query_image.empty())
      {
         drawKeypoints(pre_detect_query_image, all_query_keypoints, img);
         query_image_holder->set_image(img);
      }
   }
}

void ImageWindow::response_stats(std::vector<cv::KeyPoint>& keypoints, double& min_response, double &max_response,
                                 double& response_mean, double& response_variance, double& response_deviation)
//---------------------------------------------------------------------------------------------------------------
{
   response_mean = 0.0f;
   response_mean = std::accumulate(keypoints.begin(), keypoints.end(), 0.0f,
                                   [](double total, cv::KeyPoint kp) { return total + kp.response; });
   response_mean /= static_cast<double>(keypoints.size());

   response_variance = 0, min_response = std::numeric_limits<double>::max(),
         max_response = std::numeric_limits<double>::min();
   std::for_each (std::begin(keypoints), std::end(keypoints),
                  [response_mean, &response_variance](const cv::KeyPoint kp)
                        //--------------------------------------------------------
                  {
                     double diff = kp.response - response_mean;
                     response_variance += diff*diff;
                  });
   response_variance /= static_cast<double>(keypoints.size());
   response_deviation = sqrt(response_variance);

//   const double low_response = response_mean - 1.5f*response_deviation,
//                high_response = response_mean + 1.5f*response_deviation;
}

void ImageWindow::on_detect()
//---------------------------------
{
   setCursor(Qt::WaitCursor);
   all_train_keypoints.clear(); all_query_keypoints.clear();
   if (! all_train_descriptors.empty())
      all_train_descriptors.release();
   if (! all_query_descriptors.empty())
      all_query_descriptors.release();
   try
   {
      int top_n;
      if (! valInt(editBest, top_n, "Invalid top n"))
         top_n = -1;
      descriptor_extractor_name = feature_detector_name = featureDetectorsRadioGroup.checkedButton()->text().toStdString();
      if (! create_detector(feature_detector_name, detector))
      {
         std::stringstream errs;
         errs << "Could not create an OpenCV feature detector of type " << feature_detector_name;
         msg(errs);
         return;
      }
      ocv::detect(detector, pre_detect_train_image_bw, all_train_keypoints, all_train_descriptors, top_n);
      cv::Mat img;
      drawKeypoints(pre_detect_train_image, all_train_keypoints, img);
      train_image_holder->set_image(img);

      ocv::detect(detector, pre_detect_query_image_bw, all_query_keypoints, all_query_descriptors, top_n);
      img.release();
      drawKeypoints(pre_detect_query_image, all_query_keypoints, img);
      query_image_holder->set_image(img);

      tabs.setCurrentIndex(TRAIN_IMAGE_TAB);

      index_points.set(&all_train_keypoints);
      index.reset(new keypoint_kd_tree_t(2, index_points, nanoflann::KDTreeSingleIndexAdaptorParams(10)));
      index->buildIndex();
   }
   catch (const std::exception& e)
   {
      std::cerr << "ERROR: Exception " << e.what() << " in ImageWindow::on_detect()" << std::endl;
   }
   setCursor(Qt::ArrowCursor);
}

void ImageWindow::on_match()
//--------------------------
{
   if ( (all_train_keypoints.empty()) ||(all_query_keypoints.empty()) )
   {
      msg("No keypoints detected yet.");
      return;
   }
   setCursor(Qt::WaitCursor);
   try
   {
      cv::Mat selected_descriptors;
      std::vector<cv::KeyPoint> selected_keypoints;
      std::stringstream errs;
      matched_train_keypoints.clear();
      matched_query_keypoints.clear();
      if (! matched_train_descriptors.empty())
         matched_train_descriptors.release();
      if (! matched_query_descriptors.empty())
         matched_query_descriptors.release();
      homography_train_keypoints.clear();
      homography_query_keypoints.clear();
      if (! homography_train_descriptors.empty())
         homography_train_descriptors.release();
      if (! homography_query_descriptors.empty())
         homography_query_descriptors.release();
      matches.clear(); matched_points.clear();
      homography_matches.clear(); homography_matched_points.clear();
      for (int i = 0; i < all_train_descriptors.rows; i++)
      {
         cv::KeyPoint& kp = all_train_keypoints[i];
         KeyPointExtra_t kpc;
         kpc.class_id = static_cast<unsigned int>(kp.class_id);
         if (kpc.is_selected == 1)
         {
            selected_descriptors.push_back(all_train_descriptors.row(i));
            selected_keypoints.push_back(kp);
         }
      }
      if (selected_keypoints.empty())
      {
         msg("No keypoints selected.");
         setCursor(Qt::ArrowCursor);
         return;
      }
      cv::Ptr<cv::DescriptorMatcher> matcher;
      QString matcher_type = listItem(listMatcherType);
      if (matcher_type == "Brute Force")
      {
         int norm;
         QString matcher_norm = listItem(listMatcherNorm);
         if (matcher_norm == "HAMMING")
            norm = cv::NORM_HAMMING;
         else if (matcher_norm == "HAMMING2")
            norm = cv::NORM_HAMMING2;
         else if (matcher_norm == "L2")
            norm = cv::NORM_L2;
         else if (matcher_norm == "L2SQR")
            norm = cv::NORM_L2SQR;
         else if (matcher_norm == "L1")
            norm = cv::NORM_L1;
         else
            norm = cv::NORM_L2;
         matcher = cv::Ptr<cv::DescriptorMatcher>(new cv::BFMatcher(norm));
      }
      else
      {
         if (matcher_type == "FLANN KDTree")
            matcher = cv::Ptr<cv::FlannBasedMatcher>(new cv::FlannBasedMatcher(new cv::flann::KDTreeIndexParams(16)));
         else if (matcher_type == "FLANN KMeans")
            matcher = cv::Ptr<cv::FlannBasedMatcher>(new cv::FlannBasedMatcher(new cv::flann::KMeansIndexParams(32,
                                                                                                                -1,
                                                                                                                cvflann::CENTERS_KMEANSPP)));
         else if (matcher_type == "FLANN Composite")
            matcher = cv::Ptr<cv::FlannBasedMatcher>(
                  new cv::FlannBasedMatcher(new cv::flann::CompositeIndexParams(16, 32,
                                                                                -1, cvflann::CENTERS_KMEANSPP)));
         else if (matcher_type == "FLANN Auto")
            matcher = cv::Ptr<cv::FlannBasedMatcher>(new cv::FlannBasedMatcher(new cv::flann::AutotunedIndexParams));
      }

      cv::Mat show_matches;
      errs.str("Match error or not enough matches.");
      if ((!ocv::match(matcher, selected_keypoints, all_query_keypoints, selected_descriptors, all_query_descriptors,
                       matched_train_keypoints, matched_query_keypoints,
                       matched_train_descriptors, matched_query_descriptors, matches, &matched_points, &errs)) || (matches.size() < 4))
      {
         errs << " (No. of matches " << matches.size() << ")";
         msg(errs);
         match_image_holder->set_image(ocv::draw_matches(pre_detect_train_image, matched_train_keypoints,
                                                         pre_detect_query_image, matched_query_keypoints,
                                                         matches, show_matches));
         setCursor(Qt::ArrowCursor);
         return;
      }
      QString homography_meth = listItem(listHomographyMethod);
      int method;
      if (homography_meth.startsWith(tr("RANSAC")))
         method = cv::RANSAC;
      else if (homography_meth.startsWith(tr("RHO")))
         method = cv::RHO;
      else if (homography_meth.startsWith(tr("LMEDS")))
         method = cv::LMEDS;
      else if (homography_meth.startsWith(tr("Least")))
         method = 0;
      else method = cv::RHO;
      double reproj_error;
      if (!valDouble(editHomographyReprojErr, reproj_error, "Invalid reprojection error. Defaulting to 3"))
         reproj_error = 3;
      cv::Mat homography;
      std::vector<cv::DMatch> homographyMatches;
      errs.str("Homography error or not enough matches in homography");
      if ((!ocv::homography(matched_train_keypoints, matched_query_keypoints, matches, homography, homographyMatches,
                            &homography_matched_points, method, reproj_error)) || (homographyMatches.size() < 3))
      {
         msg(errs);
         match_image_holder->set_image(ocv::draw_matches(pre_detect_train_image, matched_train_keypoints,
                                                         pre_detect_query_image, matched_query_keypoints,
                                                         matches, show_matches));
         setCursor(Qt::ArrowCursor);
         return;
      }
//      matches = std::move(homographyMatches);

      std::for_each(std::begin(homographyMatches), std::end(homographyMatches),
                    [this](const cv::DMatch m)
                          //--------------------------------------------------------
                    {
                       homography_train_keypoints.push_back(matched_train_keypoints[m.trainIdx]);
                       homography_query_keypoints.push_back(matched_query_keypoints[m.queryIdx]);
//                       std::cout << matched_train_keypoints[m.trainIdx].pt << " | " << matched_query_keypoints[m.queryIdx].pt << std::endl;
                       homography_train_descriptors.push_back(matched_train_descriptors.row(m.trainIdx));
                       homography_query_descriptors.push_back(matched_query_descriptors.row(m.queryIdx));
                       cv::DMatch mm(static_cast<int>(homography_train_keypoints.size() - 1),
                                     static_cast<int>(homography_query_keypoints.size() - 1),
                                     m.distance);
                       homography_matches.push_back(mm);
                    });

      std::vector<cv::Point2f> train_convex_hull_pts, query_convex_hull_pts;
      cv::Point2f centroid;
      cv::RotatedRect train_rbb, query_rbb;
      cv::Rect2f train_bb, query_bb;
      errs.str("");
      if ((!ocv::convex_hull(homography_train_keypoints, train_convex_hull_pts, centroid, train_rbb, train_bb, nullptr,
                             &errs)) || (train_convex_hull_pts.size() < 4))
      {
         msg(errs);
         match_image_holder->set_image(ocv::draw_matches(pre_detect_train_image, matched_train_keypoints,
                                                         pre_detect_query_image, matched_query_keypoints,
                                                         matches, show_matches));
         setCursor(Qt::ArrowCursor);
         return;
      }
      if ((!ocv::convex_hull(homography_query_keypoints, query_convex_hull_pts, centroid, query_rbb, query_bb, nullptr,
                             &errs)) || (query_convex_hull_pts.size() < 4))
      {
         msg(errs);
         match_image_holder->set_image(ocv::draw_matches(pre_detect_train_image, matched_train_keypoints,
                                                         pre_detect_query_image, matched_query_keypoints,
                                                         matches, show_matches));
         setCursor(Qt::ArrowCursor);
         return;
      }

      cv::Mat train_image, query_image;
      pre_detect_train_image.copyTo(train_image);
      cv::rectangle(train_image, train_bb, red, 2);
      pre_detect_query_image.copyTo(query_image);
      cv::rectangle(query_image, query_bb, red, 2);
      match_image_holder->set_image(ocv::draw_matches(train_image, homography_train_keypoints,
                                                      query_image, homography_query_keypoints,
                                                      homography_matches, show_matches));
//      cv::imwrite("rect1.png", train_image);
//      cv::imwrite("rect2.png", query_image);

      //   std::vector<cv::Point2f> obj_corners(4);
      //   obj_corners[0] = cv::Point2f(train_bb.x, train_bb.y);
      //   obj_corners[1] = cv::Point2f(train_bb.x + train_bb.width, train_bb.y );
      //   obj_corners[2] = cv::Point2f(train_bb.x + train_bb.width, train_bb.y + train_bb.height);
      //   obj_corners[3] = cv::Point2f(train_bb.x, train_bb.y + train_bb.height);
      //   std::vector<cv::Point2f> scene_corners(4);
      //   cv::perspectiveTransform( obj_corners, scene_corners, homography);
      //   cv::Mat img;
      //   pre_detect_query_image.copyTo(img);
      //   cv::rectangle(img, scene_corners[0], scene_corners[2], red, 2);
      //   cv::imwrite("rect3.png", img);
   }
   catch (const std::exception& e)
   {
      std::cerr << "ERROR: Exception " << e.what() << " in ImageWindow::on_detect()" << std::endl;
   }
   setCursor(Qt::ArrowCursor);
}

void ImageWindow::on_delete_unselected()
//--------------------------------------
{
   std::vector<cv::KeyPoint> selected_keypoints;
   for (cv::KeyPoint& kp : all_train_keypoints)
   {
      KeyPointExtra_t kpc;
      kpc.class_id = static_cast<unsigned int>(kp.class_id);
      if (kpc.is_selected)
         selected_keypoints.emplace_back(kp);
   }
   if (! selected_keypoints.empty())
      all_train_keypoints = std::move(selected_keypoints);
   cv::Mat img;
   drawKeypoints(pre_detect_train_image, all_train_keypoints, img);
   train_image_holder->set_image(img);
   index_points.set(&all_train_keypoints);
   index.reset(new keypoint_kd_tree_t(2, index_points, nanoflann::KDTreeSingleIndexAdaptorParams(10)));
   index->buildIndex();
}

void ImageWindow::on_clear_selection()
//------------------------------------
{
   for (cv::KeyPoint& kp : all_train_keypoints)
   {
      KeyPointExtra_t kpc;
      kpc.class_id = static_cast<unsigned int>(kp.class_id);
      kpc.is_selected = 0;
      kp.class_id = kpc.class_id;
   }
   cv::Mat img;
   drawKeypoints(pre_detect_train_image, all_train_keypoints, img);
   train_image_holder->set_image(img);
}

void ImageWindow::on_selected_from_matched()
//-------------------------------------------
{
   if (matched_train_keypoints.empty())
   {
      msg("No matched keypoints or match not yet performed");
      return;
   }
   select_from(all_train_keypoints, matched_train_keypoints);
   cv::Mat img;
   drawKeypoints(pre_detect_train_image, all_train_keypoints, img);
   train_image_holder->set_image(img);
   select_from(all_query_keypoints, matched_query_keypoints);
   drawKeypoints(pre_detect_query_image, all_query_keypoints, img);
   query_image_holder->set_image(img);
}

void ImageWindow::on_selected_from_homography()
//-------------------------------------------
{
   if (homography_train_keypoints.empty())
   {
      msg("No homography keypoints or match not yet performed");
      return;
   }
   select_from(all_train_keypoints, homography_train_keypoints);
   cv::Mat img;
   drawKeypoints(pre_detect_train_image, all_train_keypoints, img);
   train_image_holder->set_image(img);
   select_from(all_query_keypoints, homography_query_keypoints);
   drawKeypoints(pre_detect_query_image, all_query_keypoints, img);
   query_image_holder->set_image(img);
}

void ImageWindow::select_from(std::vector<cv::KeyPoint>& all_keypoints, std::vector<cv::KeyPoint>& keypoints)
//------------------------------------------------------------------------------------------------------------
{
   if (keypoints.empty())
   {
      msg("No matched keypoints or match not yet performed");
      return;
   }
   for (cv::KeyPoint& kp : all_keypoints)
   {
      KeyPointExtra_t kpc;
      kpc.class_id = static_cast<unsigned int>(kp.class_id);
      kpc.is_selected = 0;
      kp.class_id = kpc.class_id;
   }
   for (cv::KeyPoint& kp : keypoints)
   {
      cv::KeyPoint* pkp;
      float distance = std::numeric_limits<float>::max();
      if ( (closest_keypoint(kp.pt.x, kp.pt.y, 1, pkp, distance)) && (mut::near_zero(distance, 0.01f)) )
      {
         KeyPointExtra_t kpc;
         kpc.class_id = static_cast<unsigned int>(pkp->class_id);
         kpc.is_selected = 1;
         pkp->class_id = kpc.class_id;
      }
   }
}


inline size_t match_combinations(const std::vector<std::pair<cv::Point3d, cv::Point3d>>& v, int r,
                                 std::vector<std::vector<std::pair<cv::Point3d, cv::Point3d>>>& combinations)
//------------------------------------------------------------------------------------------------------------------------------
{
   int n = static_cast<int>(v.size());
   if (n == 0) return 0;
   std::vector<bool> b(n);
   std::fill(b.begin(), b.begin() + r, true);

   do
   {
      std::vector<std::pair<cv::Point3d, cv::Point3d>> combination;
      for (int i = 0; i < n; ++i)
      {
         if (b[i])
            combination.push_back(v[i]);
      }
      combinations.push_back(combination);
   } while (std::prev_permutation(b.begin(), b.end()));
   return combinations.size();
}

size_t ImageWindow::load_3D_points(std::vector<std::pair<cv::Point3d, cv::Point3d>>& pts)
//---------------------------------------------------------------------------------------
{
   if (loaded_threeD_matches.empty())
   {
      for (const std::tuple<cv::Point3d, cv::KeyPoint, cv::Mat>& tt : threeD_matches)
      {
         const cv::Point3d& pt3d = std::get<0>(tt);
         const cv::Point2f& tpt2d = std::get<1>(tt).pt;
         size_t j = find_image_point(matched_points, cv::Point3d(tpt2d.x, tpt2d.y, 0));
         if (j > matched_points.size()) continue;
         const cv::Point3d& qpt = matched_points[j].second;
         pts.emplace_back(std::make_pair(pt3d, qpt));
      }
   }
   else
   {
      for (const std::pair<cv::Point3d, cv::Point2d> pp : loaded_threeD_matches)
      {
         const cv::Point3d& pt3d = pp.first;
         const cv::Point2d& pt2d = pp.second;
         pts.emplace_back(std::make_pair(pt3d, cv::Point3d(pt2d.x, pt2d.y, 1)));
      }
   }
   return pts.size();
}

void ImageWindow::on_pose()
//-------------------------
{
   if ( (! calibration) && (! train_image_meta.calibration) )
   {
      msg("No camera calibration. Cannot calculate pose");
      return;
   }
   QString src = poseSourceRadioGroup.checkedButton()->text();
   std::vector<std::pair<cv::Point3d, cv::Point3d>> pts, *pts_ptr;

   cv::Mat intrinsics = ( (train_image_meta.calibration) ? train_image_meta.calibration->asMat3x3d() : calibration->asMat3x3d() );
   Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> K((double *) intrinsics.data);
   Eigen::Quaterniond Q;
   Eigen::Vector3d T;
   QString method = listItem(listPoseMethod);
   std::unique_ptr<Pose> pose;
   double depth;
   Eigen::Vector3d train_g, query_g;
   int sample_size;
   if (method.startsWith("Gravity2Gravity"))
   {
      if ( (train_image_meta.correctedGravity.empty()) || (query_image_meta.correctedGravity.empty()) )
      {
         msg("Missing gravity vectors in image meta file or no image meta file");
         return;
      }
      for (int col=0; col<3; col++)
      {
         train_g[col] = train_image_meta.correctedGravity.at<double>(0, col);
         query_g[col] = query_image_meta.correctedGravity.at<double>(0, col);
      }
      depth = train_image_meta.depth;
      if ( (std::isnan(depth)) || (depth <= 0) )
         pose.reset(new Gravity2Gravity2DPose(intrinsics, train_g, query_g));
      else
         pose.reset(new Gravity2Gravity2DDepthPose(intrinsics, train_g, query_g, depth));
      if (src == "Matches")
         pts_ptr = &matched_points;
      else// if (src == "Homography")
         pts_ptr = &homography_matched_points;
//      pts.clear();
//      for (std::pair<cv::Point3d, cv::Point3d>& pp : *pts_ptr)
//      {
//         cv::Point3d& first = pp.first;
//         cv::Point3d& second = pp.second;
//         pts.emplace_back(std::make_pair(cv::Point3d(std::round(first.x), std::round(first.y), std::round(first.z)),
//                                         cv::Point3d(std::round(second.x), std::round(second.y), std::round(second.z))));
//
//      }
//      pts_ptr = &pts;
      sample_size = 3;
   }
   else if (method.startsWith("3D Gravity2Gravity"))
   {
      if ( (threeD_matches.empty()) && (loaded_threeD_matches.empty()) )
      {
         msg("No 3D points specified yet. Right click on keypoints to assign 3D coordinates to them");
         return;
      }
      for (int col=0; col<3; col++)
      {
         train_g[col] = train_image_meta.correctedGravity.at<double>(0, col);
         query_g[col] = query_image_meta.correctedGravity.at<double>(0, col);
      }
      pose.reset(new Gravity2Gravity3DPose(intrinsics, train_g, query_g));
      if (load_3D_points(pts) < 3)
      {
         msg("Not enough 3D points specified");
         return;
      }
      pts_ptr = &pts;
      sample_size = 3;
   }
   else if (method.startsWith("OpenCV PnP"))
   {
      if ( (threeD_matches.empty()) && (loaded_threeD_matches.empty()) )
      {
         msg("No 3D points specified yet. Right click on keypoints to assign 3D coordinates to them");
         return;
      }
      pose.reset(new PnP3DPose(intrinsics));
      if (load_3D_points(pts) < 4)
      {
         msg("Not enough 3D points specified");
         return;
      }
      pts_ptr = &pts;
      sample_size = 4;
   }

   double confidence;
   if (chkPoseUseRANSAC->isChecked())
   {
      double error;
      if (! valDouble(editPoseRANSACError, error, "Invalid Pose RANSAC threshold error. Defaulting to 9"))
         error = 9;
#ifdef USE_THEIA_RANSAC
      theia::RansacParameters RANSAC_parameter;
      RANSAC_parameter.error_thresh = error;
      RANSAC_parameter.max_iterations = 10000;
#else
      templransac::RANSACParams RANSAC_parameter(error);
      RANSAC_parameter.error_threshold = error;
#endif
      confidence = pose->pose_ransac(*pts_ptr, sample_size, &RANSAC_parameter);
   }
   else if (pts_ptr->size() < 15)
   {
      std::vector<std::vector<std::pair<cv::Point3d, cv::Point3d>>> combinations;
      match_combinations(*pts_ptr, sample_size, combinations);
      double min_mean_error = 9999999;
      size_t index = combinations.size();
      for (size_t j=0; j<combinations.size(); j++)
      {
         const std::vector<std::pair<cv::Point3d, cv::Point3d>>& cpts = combinations[j];
         pose->pose(cpts);
         double maxError, meanError;
         cv::Mat img;
         pose->reproject(pre_detect_query_image, cpts, maxError, meanError, img);
         if (meanError < min_mean_error)
         {
            min_mean_error = meanError;
            index = j;
         }
      }
      const std::vector<std::pair<cv::Point3d, cv::Point3d>>& cpts = combinations[index];
      pose->pose(cpts);
//      pts = std::move(cpts);
//      pts_ptr = &pts;
      confidence = 1;
   }
   else
   {
      msg("Too many points for Non-RANSAC");
      confidence = 0;
   }
   if (confidence > 0.2)
   {
      Eigen::Quaterniond Q;
      Eigen::Vector3d t;
      for (size_t no = 0; no < pose->result_count(); no++)
      {
         pose->result(no, Q, t);
         double maxError, meanError;
         cv::Mat img;
         pose->reproject(pre_detect_query_image, *pts_ptr, maxError, meanError, img);
         cv::imwrite("reproject.png", img);
         set_pose_result(no, t, Q, maxError, meanError, &img);
      }
   }
   else
   {
      Eigen::Quaterniond Q(0, 0, 0, 0);
      Eigen::Vector3d t(0, 0, 0);
      set_pose_result(0, t, Q, 9999999, 9999999, nullptr);
   }

//   double error;
//   if (! valDouble(editPoseRANSACError, error, "Invalid Pose RANSAC threshold error. Defaulting to 9"))
//      error = 9;
//#ifdef USE_THEIA_RANSAC
//   theia::RansacParameters RANSAC_parameter;
//   RANSAC_parameter.error_thresh = error;
//   RANSAC_parameter.max_iterations = 10000;
//#else
//   templransac::RANSACParams RANSAC_parameter(error);
//   RANSAC_parameter.error_threshold = error;
//#endif
//   for (double d = 2.5; d < 3.5; d += 0.01)
//   {
//      pose.reset(new Gravity2Gravity2DDepthPose(intrinsics, train_g, query_g, d));
//      pose->pose_ransac(*pts_ptr, 3, &RANSAC_parameter);
//      double maxError, meanError;
//      cv::Mat img;
//      pose->reproject(pre_detect_query_image, *pts_ptr, maxError, meanError, img);
//      std::stringstream ss;
//      ss << "reproject-" << d << ".png";
//      std::cout << d << " " << maxError << ", " << meanError << std::endl;
//      cv::imwrite(ss.str(), img);
//   }
}

bool ImageWindow::closest_keypoint(float x, float y, float radius, cv::KeyPoint*& kp, float& distance)
//-----------------------------------------------------------------------------------------
{
   if (! index) return false;
   const float query_pt[2] = { x, y };
   std::vector<std::pair<size_t, float>> matches;
   nanoflann::SearchParams params;
   const size_t no = index->radiusSearch(&query_pt[0], radius, matches, params);
   if (no > 0)
   {
      distance = matches[0].second;
      kp = &index_points.get(matches[0].first);
      return true;
   }
   distance = -1;
   return false;
}

bool ImageWindow::closest_3D_point(float x, float y, float radius, std::tuple<cv::Point3d, cv::KeyPoint, cv::Mat>*& tt,
                                   float& distance)
//--------------------------------------------------------------------------------------------------------------------
{
   if (! index3d) return false;
   const float query_pt[2] = { x, y };
   std::vector<std::pair<size_t, float>> matches;
   nanoflann::SearchParams params;
   const size_t no = index3d->radiusSearch(&query_pt[0], radius, matches, params);
   if (no > 0)
   {
      distance = matches[0].second;
      tt = &index_points3d.get(matches[0].first);
      return true;
   }
   distance = -1;
   return false;
}

void ImageWindow::drawKeypoints(const cv::InputArray& image, std::vector<cv::KeyPoint>& keypoints,
                                cv::InputOutputArray& outImage, int flags,
                                std::vector<cv::KeyPoint*>* selected_pts)
//-------------------------------------------------------------------------------------------------
{
   if( !(flags & cv::DrawMatchesFlags::DRAW_OVER_OUTIMG) )
   {
      if( image.type() == CV_8UC3 )
         image.copyTo( outImage );
      else if( image.type() == CV_8UC1 )
         cvtColor( image, outImage, cv::COLOR_GRAY2BGR );
      else
      {
         std::cerr << "drawKeypoints: Invalid image format" << std::endl;
         return;
      }
   }

   std::unique_ptr<std::vector<cv::KeyPoint*>> selected_pts_ptr;
   if (selected_pts == nullptr)
   {
      selected_pts_ptr.reset(new std::vector<cv::KeyPoint*>);
      selected_pts = selected_pts_ptr.get();
   }
   cv::Scalar color = cv::Scalar(unselected_colour.blue(), unselected_colour.green(), unselected_colour.red());
   if (selected_pts->empty())
   {
      for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
      {
         cv::KeyPoint &kp = *it;
         KeyPointExtra_t kpc;
         kpc.class_id = static_cast<unsigned int>(kp.class_id);
         if (kpc.is_selected == 1)
            selected_pts->push_back(&kp);
            //         drawKeypoint(outImage, *it, red, flags);
         else
            drawKeypoint(outImage, *it, color, flags);
      }
   }
   else
   {
      for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
      {
         cv::KeyPoint &kp = *it;
         KeyPointExtra_t kpc;
         kpc.class_id = static_cast<unsigned int>(kp.class_id);
         if (kpc.is_selected == 0)
            drawKeypoint(outImage, *it, color, flags);
      }
   }
   color = cv::Scalar(selected_colour.blue(), selected_colour.green(), selected_colour.red());
   for(auto it = selected_pts->begin() ; it != selected_pts->end(); ++it )
   {
      cv::KeyPoint* pkp = *it;
      drawKeypoint(outImage, *pkp, color, flags);
   }
}

bool ImageWindow::create_detector(std::string detectorName, cv::Ptr<cv::Feature2D>& detector)
//--------------------------------------------------------------------------------------------------
{
   detector.release();
   detectorName.erase(std::remove(detectorName.begin(), detectorName.end(), '&'), detectorName.end());
   std::transform(detectorName.begin(), detectorName.end(), detectorName.begin(), ::toupper);
   if (detectorName == "ORB")
   {
      int cfeatures, edgeThreshold=31, patchSize = 31;
      float scale;
      if (! valInt(editOrbNoFeatures, cfeatures, "ORB invalid number of features")) return false;
      if (! valFloat(editOrbScale, scale, "ORB invalid scale factor")) return false;
      QString score = listItem(listOrbScore);
      int scoreType = cv::ORB::HARRIS_SCORE;
      if (score.toStdString() == "FAST")
         scoreType = cv::ORB::FAST_SCORE;
      if (! valInt(editOrbPatchSize, patchSize, "ORB invalid patch size")) return false;
      edgeThreshold = patchSize;
      QString wta = listItem(listOrbWTA);
      detector = cv::ORB::create(cfeatures, scale, 8, edgeThreshold, 0, wta.toInt(), scoreType,
                                 patchSize);
      detector_info.set("ORB", { { "nfeatures", std::to_string(cfeatures) }, { "scaleFactor", std::to_string(scale) },
                                 { "nlevels",  "8"}, { "firstLevel",  "0" }, { "WTA_K", wta.toStdString() },
                                 { "scoreType", std::to_string(scoreType) }, { "patchSize", std::to_string(patchSize) },
                                 { "edgeThreshold", std::to_string(patchSize) } });
   }
#ifdef HAVE_OPENCV_XFEATURES2D
   if (detectorName == "SIFT")
   {
      int noFeatures, layers, edge;
      float contrast, sigma;
      if (! valInt(editSiftNoFeatures, noFeatures, "SIFT invalid max no. features")) return false;
      if (! valInt(editSiftOctaveLayers, layers, "SIFT invalid octave layers")) return false;
      if (! valFloat(editSiftContrast, contrast, "SIFT invalid contrast")) return false;
      if (! valInt(editSiftEdge, edge, "Sift edge threshold")) return false;
      if (! valFloat(editSiftSigma, sigma, "SIFT \u03C3 (sigma)")) return false;
      detector = cv::xfeatures2d::SIFT::create(noFeatures, layers, contrast, edge, sigma);
      detector_info.set("SIFT", { { "nfeatures", std::to_string(noFeatures) }, { "nOctaveLayers", std::to_string(layers) },
                                  { "contrastThreshold", std::to_string(contrast) },
                                  { "edgeThreshold", std::to_string(edge) }, { "sigma", std::to_string(sigma) } });
   }
   if (detectorName == "SURF")
   {
      int hessian, octaves, octaveLayers;
      if (! valInt(editSurfThreshold, hessian, "SURF Hessian threshold")) return false;
      if (! valInt(editSurfOctaves, octaves, "SURF octaves")) return false;
      if (! valInt(editSurfOctaveLayers, octaveLayers, "SURF octave layers")) return false;
      detector = cv::xfeatures2d::SURF::create(hessian, octaves, octaveLayers, chkSurfExtended->isChecked());
      detector_info.set("SURF", { { "hessianThreshold", std::to_string(hessian) }, { "nOctaves", std::to_string(octaves) },
                                  { "nOctaveLayers", std::to_string(octaveLayers) },
                                  { "extended", std::to_string(chkSurfExtended->isChecked()) },
                                  { "upright", "false" } });
   }
#endif
   if (detectorName == "AKAZE")
   {
      float threshold;
      int descriptor, diffusivity, octaves, octaveLayers;
      std::string descriptorName = listItem(listAkazeDescriptors).toStdString();
      if (descriptorName == "DESCRIPTOR_MLDB_UPRIGHT")
         descriptor = cv::AKAZE::DESCRIPTOR_MLDB_UPRIGHT;
      else if (descriptorName == "DESCRIPTOR_KAZE")
         descriptor = cv::AKAZE::DESCRIPTOR_KAZE;
      else if (descriptorName == "DESCRIPTOR_KAZE_UPRIGHT")
         descriptor = cv::AKAZE::DESCRIPTOR_KAZE_UPRIGHT;
      else
         descriptor = cv::AKAZE::DESCRIPTOR_MLDB;
      if (! valFloat(editAkazeThreshold, threshold, "AKAZE threshold")) return false;
      if (! valInt(editAkazeOctaves, octaves, "SURF octaves")) return false;
      if (! valInt(editAkazeOctaveLayers, octaveLayers, "SURF octave layers")) return false;
      std::string diffusivityName = listItem(listAkazeDiffusivity).toStdString();
      if (diffusivityName == "DIFF_PM_G1")
         diffusivity = cv::KAZE::DIFF_PM_G1;
      else if (diffusivityName == "DIFF_WEICKERT")
         diffusivity = cv::KAZE::DIFF_WEICKERT;
      else if (diffusivityName == "DIFF_CHARBONNIER")
         diffusivity = cv::KAZE::DIFF_CHARBONNIER;
      else
         diffusivity = cv::KAZE::DIFF_PM_G2;
      detector = cv::AKAZE::create(descriptor, 0, 3, threshold, octaves, octaveLayers, diffusivity);
      detector_info.set("AKAZE", { { "descriptor_type", std::to_string(descriptor) }, { "descriptor_size", "0" },
                                   { "descriptor_channels",  "3"}, { "threshold",  std::to_string(threshold) },
                                   { "nOctaves", std::to_string(octaves) },
                                   { "nOctaveLayers", std::to_string(octaveLayers) },
                                   { "diffusivity", std::to_string(diffusivity) } });
   }
   if (detectorName == "BRISK")
   {
      int threshold, octaves;
      float scale;
      if (! valInt(editBriskThreshold, threshold, "BRISK invalid threshold")) return false;
      if (! valInt(editBriskOctaves, octaves, "BRISK invalid octaves")) return false;
      if (! valFloat(editBriskScale, scale, "BRISK invalid scale")) return false;
      detector = cv::BRISK::create(threshold, octaves, scale);
      detector_info.set("BRISK", { { "thresh", std::to_string(threshold) }, { "octaves", std::to_string(octaves) },
                                   { "patternScale",  std::to_string(scale)} });
   }
   return (! detector.empty());
}

bool ImageWindow::load_train_img(std::string imagepath, std::stringstream *errs)
//----------------------------------------------------------------------------------
{
   if (load_image(imagepath, pre_detect_train_image, pre_detect_train_image_bw,
                  train_image_holder, train_image_meta, errs))
   {
      all_train_keypoints.clear();
      all_train_descriptors.release();
      index.reset(nullptr);
//      set_pose_image(pre_detect_train_image, train_image_label);
//      cv::Mat pose_img(pre_detect_train_image.rows, pre_detect_train_image.cols, CV_8UC3, cv::Scalar(0, 0, 0));
//      set_pose_image(pose_img, pose_image_label);
      return true;
   }
   return false;
}

bool ImageWindow::load_query_img(std::string imagepath, std::stringstream *errs)
//----------------------------------------------------------------------------------
{
   if (load_image(imagepath, pre_detect_query_image, pre_detect_query_image_bw,
                  query_image_holder, query_image_meta, errs))
   {
      all_query_keypoints.clear();
      all_query_descriptors.release();
//      set_pose_image(pre_detect_query_image, query_image_label);
      return true;
   }
   return false;
}

bool ImageWindow::load_image(const std::string imagepath, cv::Mat& pre_color_img, cv::Mat& pre_mono_image,
                             CVQtScrollableImage* image_holder, imeta::ImageMeta& image_meta, std::stringstream* errs,
                             std::string imagemeta_path)
//------------------------------------------------------------------------------------------------------------------
{
   cv::Mat img = cv::imread(imagepath, cv::IMREAD_COLOR);
   std::unique_ptr<std::stringstream> perrs;
   if (errs == nullptr)
   {
      perrs.reset(new std::stringstream);
      errs = perrs.get();
   }
   if (img.empty())
   {
      *errs << "Error reading image from " << imagepath;
      std::cerr << errs->str() << std::endl;
      return false;
   }
   pre_color_img = img;
   cv::cvtColor(img, pre_mono_image, CV_BGR2GRAY);
   image_holder->set_image(img); // creates own copy

   if (! imagemeta_path.empty())
   {
      if (image_meta.open(imagemeta_path))
         return true;
   }
   if (! image_meta.open(imagepath))
   {
      *errs << "WARNING: Could not find/open a image metafile (yaml/xml)for image " << imagepath << std::endl
            << image_meta.messages();
      std::cerr << errs->str() << std::endl;
//      return false;
   }
   return true;
}

void ImageWindow::drawKeypoint(cv::InputOutputArray img, const cv::KeyPoint& p, const cv::Scalar& color, int flags)
//------------------------------------------------------------------------------------------------------------------
{
   const int draw_shift_bits = 4;
   const int draw_multiplier = 1 << draw_shift_bits;
   cv::Point center( cvRound(p.pt.x * draw_multiplier), cvRound(p.pt.y * draw_multiplier) );

   if( flags & cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS )
   {
      int radius = cvRound(p.size/2 * draw_multiplier); // KeyPoint::size is a diameter

      // draw the circles around keypoints with the keypoints size
      circle( img, center, radius, color, 1, cv::LINE_AA, draw_shift_bits );

      // draw orientation of the keypoint, if it is applicable
      if( p.angle != -1 )
      {
         float srcAngleRad = p.angle*(float)CV_PI/180.f;
         cv::Point orient( cvRound(cos(srcAngleRad)*radius ),
                           cvRound(sin(srcAngleRad)*radius )
         );
         line( img, center, center+orient, color, 1, cv::LINE_AA, draw_shift_bits );
      }
   }
   else
   {
      // draw center with R=3
      int radius = 3 * draw_multiplier;
      circle( img, center, radius, color, 1, cv::LINE_AA, draw_shift_bits );
   }
}

void ImageWindow::setup_features_control(QLayout *layout)
//------------------------------------------------------
{
   int buttonNo = 0;
   QFrame* frame = new QFrame(this);
   QHBoxLayout* frame_layout = new QHBoxLayout(this);
   orbRadio = new QRadioButton(tr("&ORB"), this);
   connect(orbRadio,SIGNAL(clicked()),this,SLOT(onOrbSelected()));
   orbRadio->setChecked(true);
   frame_layout->addWidget(orbRadio);
   featureDetectorsRadioGroup.addButton(orbRadio, buttonNo++);
   frame_layout->addSpacing(4);
   editOrbNoFeatures = new QLineEdit("1000", this);
   editOrbNoFeatures->setValidator(new QIntValidator);
   frame_layout->addWidget(new QLabel("Max Features:")); frame_layout->addWidget(editOrbNoFeatures);
   frame_layout->addSpacing(4);
   editOrbScale = new QLineEdit("1.1", this);
   editOrbScale->setValidator(new QDoubleValidator);
   frame_layout->addWidget(new QLabel("Scale Factor:")); frame_layout->addWidget(editOrbScale);
   frame_layout->addSpacing(4);
   listOrbWTA = new QListWidget(this);
   listOrbWTA->setSelectionBehavior(QAbstractItemView::SelectionBehavior::SelectRows);
   listOrbWTA->setSelectionMode(QAbstractItemView::SelectionMode::SingleSelection);
   listOrbWTA->setFixedHeight(QFontMetrics(listOrbWTA->font()).height()*4 + 5);
   new QListWidgetItem(tr("2"), listOrbWTA);
   new QListWidgetItem(tr("3"), listOrbWTA);
   new QListWidgetItem(tr("4"), listOrbWTA);
   listOrbWTA->item(0)->setSelected(true);
   frame_layout->addWidget(new QLabel("WTA:")); frame_layout->addWidget(listOrbWTA);
   frame_layout->addSpacing(4);
   listOrbScore = new QListWidget(this);
   listOrbScore->setSelectionBehavior(QAbstractItemView::SelectionBehavior::SelectRows);
   listOrbScore->setFixedHeight(QFontMetrics(listOrbScore->font()).height()*3);
   new QListWidgetItem(tr("HARRIS"), listOrbScore);
   new QListWidgetItem(tr("FAST"), listOrbScore);
   listOrbScore->item(0)->setSelected(true);
   frame_layout->addWidget(new QLabel("Score")); frame_layout->addWidget(listOrbScore);
   frame_layout->addSpacing(4);
   editOrbPatchSize = new QLineEdit("31", this);
   editOrbPatchSize->setValidator(new QIntValidator);
   frame_layout->addWidget(new QLabel("Patch Size:")); frame_layout->addWidget(editOrbPatchSize);
   frame->setLayout(frame_layout);
   frame->updateGeometry();
   layout->addWidget(frame);

#ifdef HAVE_OPENCV_XFEATURES2D
   frame = new QFrame(this);
   frame_layout = new QHBoxLayout(this);
   siftRadio = new QRadioButton("&SIFT", this);
   connect(siftRadio,SIGNAL(clicked()),this,SLOT(onSiftSelected()));
   frame_layout->addWidget(siftRadio);
   featureDetectorsRadioGroup.addButton(siftRadio, buttonNo++);
   frame_layout->addSpacing(4);
   editSiftNoFeatures = new QLineEdit("1000", this);
   editSiftNoFeatures->setValidator(new QIntValidator);
   frame_layout->addWidget(new QLabel("Max Features:")); frame_layout->addWidget(editSiftNoFeatures);
   frame_layout->addSpacing(4);
   editSiftOctaveLayers = new QLineEdit("3", this);
   editSiftOctaveLayers->setValidator(new QIntValidator);
   frame_layout->addWidget(new QLabel("Octave Layers:")); frame_layout->addWidget(editSiftOctaveLayers);
   frame_layout->addSpacing(4);
   editSiftContrast = new QLineEdit("0.04", this);
   editSiftContrast->setValidator(new QDoubleValidator);
   frame_layout->addWidget(new QLabel("Contr. Thresh:")); frame_layout->addWidget(editSiftContrast);
   frame_layout->addSpacing(4);
   editSiftEdge = new QLineEdit("10", this);
   editSiftEdge->setValidator(new QIntValidator);
   frame_layout->addWidget(new QLabel("Edge Thresh:")); frame_layout->addWidget(editSiftEdge);
   frame_layout->addSpacing(4);
   editSiftSigma = new QLineEdit("1.6", this);
   editSiftSigma->setValidator(new QDoubleValidator);
   frame_layout->addWidget(new QLabel("\u03C3")); frame_layout->addWidget(editSiftSigma);
   frame->setLayout(frame_layout);
   frame->updateGeometry();
   layout->addWidget(frame);

   frame = new QFrame(this);
   frame_layout = new QHBoxLayout(this);
   surfRadio = new QRadioButton("S&URF", this);
   connect(surfRadio,SIGNAL(clicked()),this,SLOT(onSurfSelected()));
   frame_layout->addWidget(surfRadio);
   featureDetectorsRadioGroup.addButton(surfRadio, buttonNo++);
   frame_layout->addSpacing(4);
   editSurfThreshold = new QLineEdit("100", this);
   editSurfThreshold->setValidator(new QIntValidator);
   frame_layout->addWidget(new QLabel("Hessian Thresh:")); frame_layout->addWidget(editSurfThreshold);
   frame_layout->addSpacing(4);
   editSurfOctaves = new QLineEdit("4", this);
   editSurfOctaves->setValidator(new QIntValidator);
   frame_layout->addWidget(new QLabel("Octaves:")); frame_layout->addWidget(editSurfOctaves);
   frame_layout->addSpacing(4);
   editSurfOctaveLayers = new QLineEdit("3", this);
   editSurfOctaveLayers->setValidator(new QIntValidator);
   frame_layout->addWidget(new QLabel("Octave Layers:")); frame_layout->addWidget(editSurfOctaveLayers);
   frame_layout->addSpacing(4);
   chkSurfExtended = new QCheckBox("Ext. Desc", this);
   chkSurfExtended->setChecked(false);
   frame_layout->addWidget(chkSurfExtended);
   frame->setLayout(frame_layout);
   frame->updateGeometry();
   layout->addWidget(frame);
#endif

   frame = new QFrame(this);
   frame_layout = new QHBoxLayout(this);
   akazeRadio = new QRadioButton("&Akaze", this);
   connect(akazeRadio,SIGNAL(clicked()),this,SLOT(onAkazeSelected()));
   frame_layout->addWidget(akazeRadio);
   featureDetectorsRadioGroup.addButton(akazeRadio, buttonNo++);
   frame_layout->addSpacing(4);
   listAkazeDescriptors = new QListWidget(this);
   listAkazeDescriptors->setSelectionBehavior(QAbstractItemView::SelectionBehavior::SelectRows);
   listAkazeDescriptors->setFixedHeight(QFontMetrics(listAkazeDescriptors->font()).height()*5 + 5);
   frame_layout->addWidget(new QLabel("Descriptor:"));
   new QListWidgetItem(tr("DESCRIPTOR_MLDB"), listAkazeDescriptors);
   new QListWidgetItem(tr("DESCRIPTOR_MLDB_UPRIGHT"), listAkazeDescriptors);
   new QListWidgetItem(tr("DESCRIPTOR_KAZE"), listAkazeDescriptors);
   new QListWidgetItem(tr("DESCRIPTOR_KAZE_UPRIGHT"), listAkazeDescriptors);
   listAkazeDescriptors->item(0)->setSelected(true);
   frame_layout->addWidget(listAkazeDescriptors);
   frame_layout->addSpacing(4);
   frame_layout->addWidget(new QLabel("Threshold:"));
   editAkazeThreshold = new QLineEdit("0.001", this);
   editAkazeThreshold->setValidator(new QDoubleValidator);
   frame_layout->addWidget(editAkazeThreshold);
   frame_layout->addSpacing(4);
   editAkazeOctaves = new QLineEdit("4", this);
   editAkazeOctaves->setValidator(new QIntValidator);
   frame_layout->addWidget(new QLabel("Octaves:")); frame_layout->addWidget(editAkazeOctaves);
   frame_layout->addSpacing(4);
   editAkazeOctaveLayers = new QLineEdit("4", this);
   editAkazeOctaveLayers->setValidator(new QIntValidator);
   frame_layout->addWidget(new QLabel("Octave Layers:")); frame_layout->addWidget(editAkazeOctaveLayers);
   frame_layout->addSpacing(4);
   frame_layout->addWidget(new QLabel("Diffusivity:"));
   listAkazeDiffusivity  = new QListWidget(this);
   listAkazeDiffusivity->setSelectionBehavior(QAbstractItemView::SelectionBehavior::SelectRows);
   listAkazeDiffusivity->setFixedHeight(QFontMetrics(listAkazeDiffusivity->font()).height()*5+5);
   new QListWidgetItem(tr("DIFF_PM_G2"), listAkazeDiffusivity);
   new QListWidgetItem(tr("DIFF_PM_G1"), listAkazeDiffusivity);
   new QListWidgetItem(tr("DIFF_WEICKERT"), listAkazeDiffusivity);
   new QListWidgetItem(tr("DIFF_CHARBONNIER"), listAkazeDiffusivity);
   listAkazeDiffusivity->item(0)->setSelected(true);
   frame_layout->addWidget(listAkazeDiffusivity);
   frame->setLayout(frame_layout);
   frame->updateGeometry();
   layout->addWidget(frame);

   frame = new QFrame(this);
   frame_layout = new QHBoxLayout(this);
   briskRadio = new QRadioButton("&BRISK", this);
   connect(briskRadio,SIGNAL(clicked()),this,SLOT(onBriskSelected()));
   frame_layout->addWidget(briskRadio);
   featureDetectorsRadioGroup.addButton(briskRadio, buttonNo++);
   frame_layout->addSpacing(4);
   editBriskThreshold = new QLineEdit("30", this);
   editBriskThreshold->setValidator(new QIntValidator);
   frame_layout->addWidget(new QLabel("Threshold:")); frame_layout->addWidget(editBriskThreshold);
   frame_layout->addSpacing(4);
   editBriskOctaves = new QLineEdit("3", this);
   editBriskOctaves->setValidator(new QIntValidator);
   frame_layout->addWidget(new QLabel("Octaves:")); frame_layout->addWidget(editBriskOctaves);
   frame_layout->addSpacing(4);
   editBriskScale = new QLineEdit("1.0", this);
   editBriskScale->setValidator(new QDoubleValidator);
   frame_layout->addWidget(new QLabel("Pattern Scale:")); frame_layout->addWidget(editBriskScale);
   frame->setLayout(frame_layout);
   frame->updateGeometry();
   layout->addWidget(frame);

   frame = new QFrame(this);
   frame_layout = new QHBoxLayout(this);
   frame_layout->setAlignment(Qt::AlignLeft);
   editBest = new QLineEdit("-1", this);
   editBest->setValidator(new QIntValidator);
   editBest->setFixedWidth(QFontMetrics(editBest->font()).width('X')*5);
   QLabel *label = new QLabel("Top n:");
   label->setFixedWidth(QFontMetrics(label->font()).width('X')*7+5);
   frame_layout->addWidget(label); frame_layout->addWidget(editBest);
   frame_layout->addSpacing(4);
   selectedColorButton = new QPushButton("Selected Color", this);
   QColor col = QColor(selected_colour);
   QString qss = QString("background-color: %1").arg(col.name());
   selectedColorButton->setStyleSheet(qss);
   selectedColorButton->setFixedWidth(QFontMetrics(selectedColorButton->font()).width('X')*14+5);
   connect(selectedColorButton, SIGNAL (released()), this, SLOT (on_selected_color_change()));
   frame_layout->addWidget(selectedColorButton);
   unselectedColorButton = new QPushButton("Unselected Color", this);
   col = QColor(unselected_colour);
   qss = QString("background-color: %1").arg(col.name());
   unselectedColorButton->setStyleSheet(qss);
   unselectedColorButton->setFixedWidth(QFontMetrics(selectedColorButton->font()).width('X')*15+5);
   connect(unselectedColorButton, SIGNAL (released()), this, SLOT (on_unselected_color_change()));
   frame_layout->addWidget(unselectedColorButton);
   detectButton = new QPushButton("&Detect", this);
   detectButton->setFixedWidth(QFontMetrics(detectButton->font()).width('X')*15+5);
   connect(detectButton, SIGNAL (released()), this, SLOT (on_detect()));
   frame_layout->addWidget(detectButton);
   frame->setLayout(frame_layout);
   frame->updateGeometry();
   layout->addWidget(frame);

   frame = new QFrame(this);
   frame_layout = new QHBoxLayout(this);
   frame_layout->setAlignment(Qt::AlignLeft);
   listMatcherType = new QListWidget(this);
   listMatcherType->setSelectionBehavior(QAbstractItemView::SelectionBehavior::SelectRows);
   listMatcherType->setFixedHeight(QFontMetrics(listMatcherType->font()).height()*5 + 5);
   frame_layout->addWidget(new QLabel("Matcher Type:"));
   new QListWidgetItem(tr("Brute Force"), listMatcherType);
   new QListWidgetItem(tr("FLANN KDTree"), listMatcherType);
   new QListWidgetItem(tr("FLANN KMeans"), listMatcherType);
   new QListWidgetItem(tr("FLANN Composite"), listMatcherType);
   new QListWidgetItem(tr("FLANN Auto"), listMatcherType);
   listMatcherType->item(0)->setSelected(true);
   frame_layout->addWidget(listMatcherType);
   frame_layout->addSpacing(4);
   listMatcherNorm = new QListWidget(this);
   listMatcherNorm->setSelectionBehavior(QAbstractItemView::SelectionBehavior::SelectRows);
   listMatcherNorm->setFixedHeight(QFontMetrics(listMatcherNorm->font()).height()*5 + 5);
   frame_layout->addWidget(new QLabel("Matcher Norm:"));
   frame_layout->addWidget(listMatcherNorm);
   frame_layout->addStretch(1);
   listHomographyMethod = new QListWidget(this);
   listHomographyMethod->setSelectionBehavior(QAbstractItemView::SelectionBehavior::SelectRows);
   listHomographyMethod->setFixedHeight(QFontMetrics(listHomographyMethod->font()).height()*5 + 5);
   frame_layout->addWidget(new QLabel("Homography Method:"));
   new QListWidgetItem(tr("RANSAC"), listHomographyMethod);
   new QListWidgetItem(tr("RHO (PROSAC)"), listHomographyMethod);
   new QListWidgetItem(tr("LMEDS"), listHomographyMethod);
   new QListWidgetItem(tr("Least Squares"), listHomographyMethod);
   listHomographyMethod->item(0)->setSelected(true);
   frame_layout->addWidget(listHomographyMethod);
   frame_layout->addSpacing(4);
   editHomographyReprojErr = new QLineEdit("3", this);
   editHomographyReprojErr->setValidator(new QDoubleValidator(0, 50, 3));
   editHomographyReprojErr->setFixedWidth(QFontMetrics(editBest->font()).width('X')*8);
   label = new QLabel("Reprojection error:");
   label->setFixedWidth(QFontMetrics(label->font()).width('X')*20);
   frame_layout->addWidget(label); frame_layout->addWidget(editHomographyReprojErr);
   frame_layout->addSpacing(4);

   matchButton = new QPushButton("&Match", this);
   matchButton->setFixedWidth(QFontMetrics(matchButton->font()).width('X')*7+5);
   connect(matchButton, SIGNAL (released()), this, SLOT (on_match()));
   frame_layout->addWidget(matchButton);
   setNorms();

   frame->setLayout(frame_layout);
   frame->updateGeometry();
   layout->addWidget(frame);

   frame->setLayout(frame_layout);
   frame->updateGeometry();
   layout->addWidget(frame);
}

void ImageWindow::setNorms()
//--------------------------
{
   listMatcherNorm->clear();
   if (is_bit_descriptor)
   {
      new QListWidgetItem(tr("HAMMING"), listMatcherNorm);
      new QListWidgetItem(tr("HAMMING2"), listMatcherNorm);
   }
   else
   {
      new QListWidgetItem(tr("L2"), listMatcherNorm);
      new QListWidgetItem(tr("L2SQR"), listMatcherNorm);
      new QListWidgetItem(tr("L1"), listMatcherNorm);
   }
   listMatcherNorm->item(0)->setSelected(true);
}

QWidget* ImageWindow::pose_layout()
//-----------------------------
{
   QSize size(0, 0);
   pose_widget = new QWidget(this);
   QVBoxLayout* layout = new QVBoxLayout;
   layout->setAlignment(Qt::AlignVCenter);
//   QFrame* frame = new QFrame(this);
//   pose_image_layout = new QHBoxLayout(this);
//   pose_image_layout->setAlignment(Qt::AlignLeft);
//   train_image_label = new QLabel(parentWidget());
//   train_image_label->setScaledContents(true);
//   train_image_label->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
//   pose_image_layout->addWidget(train_image_label);
//   query_image_label = new QLabel(parentWidget());
//   query_image_label->setScaledContents(true);
//   query_image_label->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
//   pose_image_layout->addWidget(query_image_label);
//   frame->setLayout(pose_image_layout);
//   frame->updateGeometry();
//   layout->addWidget(frame);

   QFrame* frame = new QFrame(this);
   QHBoxLayout* frame_layout = new QHBoxLayout(this);
   frame_layout->setAlignment(Qt::AlignLeft);
   listPoseMethod = new QListWidget(this);
   listPoseMethod->setSelectionBehavior(QAbstractItemView::SelectionBehavior::SelectRows);
   listPoseMethod->setFixedHeight(QFontMetrics(listPoseMethod->font()).height()*5 + 5);
   listPoseMethod->setFixedWidth(QFontMetrics(listPoseMethod->font()).width("X")*25 + 5);
   new QListWidgetItem(tr("Gravity2Gravity"), listPoseMethod);
   new QListWidgetItem(tr("3D Gravity2Gravity"), listPoseMethod);
   new QListWidgetItem(tr("OpenCV PnP"), listPoseMethod);
   QLabel* label = new QLabel("Pose  Method:", this);
   label->setFixedWidth(QFontMetrics(label->font()).width("X")*12 + 2);
   frame_layout->addWidget(label);
   listPoseMethod->item(0)->setSelected(true);
   frame_layout->addWidget(listPoseMethod);
   connect(listPoseMethod, SIGNAL(itemSelectionChanged()), this, SLOT(pose_method_selected()));
   label = new QLabel("Use Points from:", this);
   label->setFixedWidth(QFontMetrics(label->font()).width("X")*16 + 2);
   frame_layout->addWidget(label);
   poseFromHomography = new QRadioButton(tr("Homography"), this);
   poseFromHomography->setChecked(true);
   poseFromHomography->setFixedWidth(QFontMetrics(poseFromHomography->font()).width("X")*16 + 2);
   poseSourceRadioGroup.addButton(poseFromHomography);
   frame_layout->addWidget(poseFromHomography);
   poseFromMatched = new QRadioButton(tr("Matches"), this);
   poseFromMatched->setFixedWidth(QFontMetrics(poseFromMatched->font()).width("X")*10 + 2);
   poseSourceRadioGroup.addButton(poseFromMatched);
   frame_layout->addWidget(poseFromMatched);
   chkPoseUseRANSAC = new QCheckBox("Use RANSAC:", this);
   chkPoseUseRANSAC->setChecked(true);
   chkPoseUseRANSAC->setFixedWidth(QFontMetrics(chkPoseUseRANSAC->font()).width("X")*15 + 2);
   frame_layout->addWidget(chkPoseUseRANSAC);
   editPoseRANSACError = new QLineEdit("49", this);
   editPoseRANSACError->setValidator(new QDoubleValidator);
   editPoseRANSACError->setFixedWidth(QFontMetrics(editPoseRANSACError->font()).width("X")*10 + 2);
   label = new QLabel("RANSAC Error:", this); label->setFixedWidth(QFontMetrics(label->font()).width("X")*12 + 2);
   frame_layout->addWidget(label); frame_layout->addWidget(editPoseRANSACError);
   poseButton = new QPushButton("&Pose", this);
   poseButton->setFixedWidth(QFontMetrics(poseButton->font()).width('X')*5+5);
   connect(poseButton, SIGNAL (released()), this, SLOT (on_pose()));
   frame_layout->addWidget(poseButton);
   frame->setLayout(frame_layout);
   frame->updateGeometry();
   size += frame->size();
   layout->addWidget(frame);

   stackedPoseLayout = new QStackedLayout;
   posePageComboBox = new QComboBox;
   posePageComboBox->addItem(tr("Result 1"));
   layout->addWidget(posePageComboBox);
   posePageComboBox->setFixedWidth(QFontMetrics(posePageComboBox->font()).width("X")*15 + 2);
   layout->addLayout(stackedPoseLayout);
   connect(posePageComboBox, SIGNAL(activated(int)), stackedPoseLayout, SLOT(setCurrentIndex(int)));

   for (int i=0; i<MAX_POSE_RESULTS; i++)
      stackedPoseLayout->addWidget(addPoseResultWidget());

   pose_widget->setLayout(layout);
//   pose_widget->setMinimumSize(size);


   return pose_widget;
}

QWidget* ImageWindow::addPoseResultWidget()
//-----------------------------------------
{
   QWidget* poseResult = new QWidget(this);
   QHBoxLayout* result_layout = new QHBoxLayout(this);
   result_layout->setAlignment(Qt::AlignLeft);
   QFrame* frame = new QFrame(this);
   QFormLayout* frame_layout = new QFormLayout(this);
   QLabel* translation_label = new QLabel(this);
   translation_labels.push_back(translation_label);
   frame_layout->addRow("Translation", translation_label);
   QLabel* rotation_label = new QLabel(this);
   rotation_labels.push_back(rotation_label);
   frame_layout->addRow("Rotation:", rotation_label);
   QLabel* error_label = new QLabel(this);
   error_labels.push_back(error_label);
   frame_layout->addRow("Error:", error_label);
   frame->setLayout(frame_layout);
   result_layout->addWidget(frame);

   QLabel* pose_image_label = new QLabel(this);
   pose_image_label->setScaledContents(true);
   pose_image_label->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
   pose_image_label->setAutoFillBackground(true);
   pose_image_label->setStyleSheet("color: green;");
   pose_image_label->setMinimumSize(QSize(640, 480));
   pose_image_label->setMaximumSize(QSize(640, 480));
   QImage qimg(640, 480, QImage::Format_RGB888);
   qimg.fill(qRgba(0, 0, 0, 0));
   QPixmap pixels = QPixmap::fromImage(qimg);
   pose_image_label->setPixmap(pixels);
   pose_image_label->adjustSize();
   pose_image_labels.push_back(pose_image_label);
   result_layout->addWidget(pose_image_label);

   poseResult->setLayout(result_layout);
   QSize size(frame->width() + QFontMetrics(rotation_label->font()).width("X")*100 + 640, std::max(frame->height(), pose_image_label->height()) + 50);
   poseResult->setMinimumSize(size);
   QScrollArea *pose_scroll = new QScrollArea(this);
   pose_scroll->setWidget(poseResult);
   pose_scrolls.push_back(pose_scroll);
   return pose_scroll;
}

void ImageWindow::set_pose_result(size_t no, Eigen::Vector3d& T, Eigen::Quaterniond& Q, double maxError, double meanError,
                                  cv::Mat* image)
//----------------------------------------------------------------------------------------------------------------------
{
   if (no >= MAX_POSE_RESULTS) return;
   std::stringstream ss;
   ss  << std::fixed << std::setprecision(4) << T[0] << ", " << T[1] << ", " << T[2];
   translation_labels[no]->setText(ss.str().c_str());

   Eigen::Matrix3d R = Q.toRotationMatrix();
   const Eigen::Vector3d euler = R.eulerAngles(0, 1, 2);
   ss.str("");
   ss  << std::fixed << std::setprecision(4) << "[ " << Q.w() << ", (" << Q.x() << ", " << Q.y() << ", " << Q.z() << ") ] (Roll: "
       << mut::radiansToDegrees(euler[0]) << "\u00B0, Pitch: " << mut::radiansToDegrees(euler[1]) << "\u00B0, Yaw: "
       << mut::radiansToDegrees(euler[2]) << ")";
   rotation_labels[no]->setText(ss.str().c_str());

   ss.str("");
   ss << std::fixed << std::setprecision(4) << "Max: " << maxError << " Mean: " << meanError;
   error_labels[no]->setText(ss.str().c_str());
//   pose_image_layout->update();
//   imageLabel->setMinimumSize(QSize(image.rows, image.cols));
   if ( (image != nullptr) && (! image->empty()) )
   {
      QLabel* imageLabel = pose_image_labels[no];
      imageLabel->updateGeometry();
      cv::Mat img;
      QSize sze = imageLabel->size();
      int h = sze.height(), w = sze.width();
      cv::resize(*image, img, cv::Size(w, h), 0, 0, cv::INTER_LANCZOS4);
      cv::cvtColor(img, img, CV_BGR2RGB);
      //   cv::cvtColor(image, img, CV_BGR2RGB);
      QImage qimg(img.data, img.cols, img.rows, QImage::Format_RGB888);
      QPixmap pixels = QPixmap::fromImage(qimg);
      imageLabel->setPixmap(pixels);
      imageLabel->adjustSize();
      imageLabel->update();
      imageLabel->updateGeometry();
   }
   if (no > no_results)
   {
      no_results = no;
      posePageComboBox->clear();
      for (size_t i=0; i<=no_results; i++)
      {
         QString s("Result ");
         s = s + QString::number(i) + QString(" of ") + QString::number(no_results);
         posePageComboBox->addItem(s);
      }
   }
   pose_widget->updateGeometry();
   pose_widget->update();
}

void ImageWindow::pose_method_selected()
//--------------------------------------
{
   QString s = listItem(listPoseMethod);
   if (s == "Gravity2Gravity")
   {
      if (homography_train_keypoints.size() < 3)
         poseFromHomography->setEnabled(false);
      else
         poseFromHomography->setEnabled(true);
      if (matched_train_keypoints.size() < 3)
         poseFromMatched->setEnabled(false);
      else
         poseFromMatched->setEnabled(true);
   }
   if (s.startsWith("OpenCV PnP"))
   {
      if (homography_train_keypoints.size() < 4)
         poseFromHomography->setEnabled(false);
      else
         poseFromHomography->setEnabled(true);
      if (matched_train_keypoints.size() < 4)
         poseFromMatched->setEnabled(false);
      else
         poseFromMatched->setEnabled(true);
   }
}

void ImageWindow::set_pose_image(const cv::Mat& image, QLabel* imageLabel)
//------------------------------------------------------------------------
{
   pose_widget->updateGeometry();
//   pose_image_layout->update();
//   imageLabel->setMinimumSize(QSize(image.rows, image.cols));
   imageLabel->updateGeometry();
   cv::Mat img;
   QSize sze = imageLabel->size();
   int h = sze.height(), w = sze.width();
   cv::resize(image, img, cv::Size(w, h), 0, 0, cv::INTER_LANCZOS4);
   cv::cvtColor(img, img, CV_BGR2RGB);
//   cv::cvtColor(image, img, CV_BGR2RGB);
   QImage qimg(img.data, img.cols, img.rows, QImage::Format_RGB888);
   QPixmap pixels = QPixmap::fromImage(qimg);
   imageLabel->setPixmap(pixels);
   imageLabel->adjustSize();
   pose_widget->updateGeometry();
   pose_widget->update();
//   imageLabel->update();
}