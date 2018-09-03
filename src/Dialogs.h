#ifndef _DIALOGS_H_6920d1c0_3033_44dd_a1f9_3b4d1ffe7628
#define _DIALOGS_H_6920d1c0_3033_44dd_a1f9_3b4d1ffe7628

#include <string>
#ifdef FILESYSTEM_EXPERIMENTAL
#include <experimental/filesystem>
namespace filesystem = std::experimental::filesystem;
#elif defined(STD_FILESYSTEM)
#include <filesystem>
namespace filesystem = std::filesystem;
#else
#include <boost/filesystem>
#endif

#include <QDialog>
#include <QWidget>
#include <QString>
#include <QLabel>
#include <QLineEdit>

#include <opencv2/core/core.hpp>

class SaveDialog : public QDialog
//================================
{  Q_OBJECT

public:
   explicit SaveDialog(QWidget* parent =nullptr, double depth =std::numeric_limits<double>::quiet_NaN());
   long id() { return editId->text().toLong(); }
   std::string ids() { return editId->text().toStdString(); }
   long rid() { return editRid->text().toLong(); }
   std::string rids() { return editRid->text().toStdString(); }
   std::string description() { return editDescription->text().toStdString(); }
   double latitude() { return editLatitude->text().toDouble(); }
   double longitude() { return editLongitude->text().toDouble(); }
   double altitude() { return editAltitude->text().toDouble(); }
   double depth() { return editDepth->text().toDouble(); }
   std::string path() { return labelSaveLocation->text().toStdString(); }

private:
   filesystem::path dir, filepath;
   QLineEdit  *editId, *editRid, *editDescription, *editLatitude, *editLongitude, *editAltitude, *editDepth;
   QLabel *saveDirectory, *labelSaveLocation;
   QPushButton *chooseBaseDir;

private slots:
   void on_choose_basedir();
   void onIdChange(const QString &id);
   void onRidChange(const QString &s);
   void accept() override;
   void reject() override { labelSaveLocation->setText(""); }
};

class SaveMatchDialog : public QDialog
//================================
{  Q_OBJECT

public:
   explicit SaveMatchDialog(QWidget* parent =nullptr, double depth =std::numeric_limits<double>::quiet_NaN());
   double depth() { return editDepth->text().toDouble(); }
   std::string name() { return editName->text().toStdString(); }
   std::string path() { return labelSaveLocation->text().toStdString(); }

   virtual void accept();

private:
   filesystem::path dir;
   QLineEdit  *editDirName, *editName, *editDepth;
   QLabel *saveDirectory, *labelSaveLocation;
   QPushButton *chooseBaseDir;

private slots:
   void on_choose_basedir();
   void onDirNameChange(const QString &dirname);
};

class ThreeDCoordsDialog : public QDialog
//========================================
{
Q_OBJECT

public:
   explicit ThreeDCoordsDialog(cv::KeyPoint& kp, cv::Mat* desc, std::function<void(cv::KeyPoint&, cv::Mat&, cv::Point3d&)>& callback,
                               QWidget* parent = nullptr, double z =-1, double x =std::numeric_limits<float>::quiet_NaN(),
                               double y =std::numeric_limits<float>::quiet_NaN(), bool isHorizontal =false);

   void accept() override;

private:
   std::function<void(cv::KeyPoint&, cv::Mat&, cv::Point3d&)> callback;
   QLineEdit  *x_text, *y_text, *z_text;
   cv::KeyPoint keypoint;
   cv::Mat descriptor;
   double xw, yw, zw;

};
#endif // _DIALOGS_H_6920d1c0_3033_44dd_a1f9_3b4d1ffe7628
