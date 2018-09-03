#include "Dialogs.h"

#include <cmath>

#include <QFrame>
#include <QBoxLayout>
#include <QFormLayout>
#include <QDialogButtonBox>
#include <QIntValidator>
#include <QDoubleValidator>
#include <QFileDialog>
#include <QPushButton>
#include <QMessageBox>
#include <QDoubleValidator>

#include "util.h"

SaveDialog::SaveDialog(QWidget *parent, double depth) : QDialog(parent), dir(filesystem::absolute(filesystem::current_path()))
//----------------------------------------------------------------------------------------
{
   QVBoxLayout* layout = new QVBoxLayout(this);
   QFormLayout* form_layout = new QFormLayout(this);
   saveDirectory = new QLabel(dir.c_str(), this);
   chooseBaseDir = new QPushButton("&Change", this);
   chooseBaseDir->setFixedWidth(QFontMetrics(chooseBaseDir->font()).width('X')*7+5);
   connect(chooseBaseDir, SIGNAL (released()), this, SLOT (on_choose_basedir()));
   QHBoxLayout* dirLayout = new QHBoxLayout(this);
   dirLayout->addWidget(new QLabel("Base Dir:")); dirLayout->addWidget(saveDirectory); dirLayout->addWidget(chooseBaseDir);
   form_layout->addRow("Dir", dirLayout);
   editId = new QLineEdit(this); editRid = new QLineEdit(this); editDescription = new QLineEdit(this);
   editLatitude = new QLineEdit(this); editLongitude = new QLineEdit(this); editAltitude = new QLineEdit("0", this);
   editLatitude->setValidator(new QDoubleValidator(-90, 90, 5, this));
   editLongitude->setValidator(new QDoubleValidator(-180, 180, 5, this));
   editAltitude->setValidator(new QDoubleValidator(0, 50000, 5, this));
   QIntValidator* depthValidator = new QIntValidator(this); depthValidator->setBottom(0);
   if ( (! std::isnan(depth)) && (depth > 0) )
      editDepth = new QLineEdit(QString::number(depth), this);
   else
      editDepth = new QLineEdit("-1", this);
   editDepth->setValidator(depthValidator);
   QHBoxLayout* idLayout = new QHBoxLayout(this);
   idLayout->addWidget(editId);
   QLabel* mandAsterisk = new QLabel("(Required Numeric)");
   QColor red = QColor("red");
   QString qss = QString("background-color: %1").arg(red.name());
   mandAsterisk->setStyleSheet(qss);
   idLayout->addWidget(mandAsterisk);
   form_layout->addRow(tr("Feature Id"), idLayout);
   QHBoxLayout* ridLayout = new QHBoxLayout(this);
   ridLayout->addWidget(editRid);
   mandAsterisk = new QLabel("(Required Numeric)");
   ridLayout->addWidget(mandAsterisk);
   form_layout->addRow(tr("Representation Id (within feature)"), ridLayout);
   form_layout->addRow(tr("Description"), editDescription);
   QHBoxLayout* locationLayout = new QHBoxLayout(this);
   locationLayout->addWidget(new QLabel("Latitude:")); locationLayout->addWidget(editLatitude);
   locationLayout->addSpacing(5);
   locationLayout->addWidget(new QLabel("Longitude:")); locationLayout->addWidget(editLongitude);
   locationLayout->addSpacing(5);
   locationLayout->addWidget(new QLabel("Altitude:")); locationLayout->addWidget(editAltitude);
   form_layout->addRow("Location", locationLayout);
   form_layout->addRow(tr("Depth"), editDepth);
   labelSaveLocation = new QLabel(this);
   form_layout->addRow(tr("Save to"), labelSaveLocation);
   layout->addLayout(form_layout);
   QDialogButtonBox* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
   layout->addWidget(buttons);
   setLayout(layout);
   setWindowTitle("Save");
   connect(editId, SIGNAL(textChanged(const QString &)), this, SLOT(onIdChange(const QString &)));
   connect(editRid, SIGNAL(textChanged(const QString &)), this, SLOT(onRidChange(const QString &)));
   connect(buttons, SIGNAL(accepted()), this, SLOT(accept()));
   connect(buttons, SIGNAL(rejected()), this, SLOT(reject()));
}

void SaveDialog::onIdChange(const QString &id)
//------------------------------------------------
{
   if (trim(id.toStdString()).empty())
      labelSaveLocation->setText("");
   else
   {
      std::string rid = trim(editRid->text().toStdString());
      filesystem::path p = dir / id.toStdString();
      if (!rid.empty())
         p = p / (rid + ".yaml");
      labelSaveLocation->setText(QString(filesystem::absolute(p).string().c_str()));
   }
}

void SaveDialog::onRidChange(const QString &rid)
//----------------------------------------------
{
   std::string id = trim(editId->text().toStdString());
   if (id.empty())
      labelSaveLocation->setText("");
   else
   {
      filesystem::path p = dir / id;
      if (! trim(rid.toStdString()).empty())
         p = p / rid.toStdString();
      labelSaveLocation->setText(QString(filesystem::absolute(p).string().c_str()));
   }
}

void SaveDialog::on_choose_basedir()
//------------------------------
{
   QString directory = QFileDialog::getExistingDirectory(this, tr("Select Base Directory"),
                                                         filesystem::absolute(dir).string().c_str(),
                                                         QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
   filesystem::path new_dir(directory.toStdString());
   if (! filesystem::is_directory(new_dir))
   {
      QMessageBox::warning(this, "Choose Base Directory", directory + " not a valid directory.");
      return;
   }
   dir = filesystem::absolute(new_dir);
   saveDirectory->setText(dir.c_str());
}

void SaveDialog::accept()
//-----------------------
{
   std::string id = trim(editId->text().toStdString());
   std::string rid = trim(editRid->text().toStdString());
   if ( (id.empty()) || (rid.empty()) )
   {
      QMessageBox::warning(this, "Save Object Representation", "Both id and representation id must be specified.");
      labelSaveLocation->setText("");
      setResult(0);
      hide();
      return;
   }
   setResult(1);
   hide();
}

SaveMatchDialog::SaveMatchDialog(QWidget* parent, double depth) : QDialog(parent), dir(filesystem::absolute(filesystem::current_path()))
//--------------------------------------------------------------------------------------------------------------------------------------
{
   QVBoxLayout* layout = new QVBoxLayout(this);
   QFormLayout* form_layout = new QFormLayout(this);
   saveDirectory = new QLabel(dir.c_str(), this);
   chooseBaseDir = new QPushButton("&Change", this);
   chooseBaseDir->setFixedWidth(QFontMetrics(chooseBaseDir->font()).width('X')*7+5);
   connect(chooseBaseDir, SIGNAL (released()), this, SLOT (on_choose_basedir()));
   QHBoxLayout* boxLayout = new QHBoxLayout(this);
   boxLayout->addWidget(saveDirectory); boxLayout->addWidget(chooseBaseDir);
   form_layout->addRow("Base Directory:", boxLayout);

   editDirName = new QLineEdit(this);
   form_layout->addRow("Directory (relative to base):", editDirName);
   connect(editDirName, SIGNAL(textChanged(const QString &)), this, SLOT(onDirNameChange(const QString &)));

   editName = new QLineEdit(this);
   form_layout->addRow("Match Name:", editName);

   editDepth = new QLineEdit(this);
   editDepth->setText(QString::number(depth));
   form_layout->addRow(tr("Depth"), editDepth);
   labelSaveLocation = new QLabel(this);
   form_layout->addRow(tr("Save to"), labelSaveLocation);
   layout->addLayout(form_layout);
   QDialogButtonBox* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
   layout->addWidget(buttons);
   setLayout(layout);
   connect(buttons, SIGNAL(accepted()), this, SLOT(accept()));
   connect(buttons, SIGNAL(rejected()), this, SLOT(reject()));
   setWindowTitle("Save Matches");
}

void SaveMatchDialog::on_choose_basedir()
//------------------------------
{
   QString directory = QFileDialog::getExistingDirectory(this, tr("Select Base Directory"),
                                                         filesystem::absolute(dir).string().c_str(),
                                                         QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
   filesystem::path new_dir(directory.toStdString());
   if (! filesystem::is_directory(new_dir))
   {
      QMessageBox::warning(this, "Choose Base Directory", directory + " not a valid directory.");
      return;
   }
   dir = filesystem::absolute(new_dir);
   saveDirectory->setText(dir.c_str());
}

void SaveMatchDialog::onDirNameChange(const QString& dirname)
//-----------------------------------------------------------
{
   std::string name = trim(dirname.toStdString());
   if (name.empty())
      labelSaveLocation->setText("");
   else
   {
      filesystem::path p = dir / name;
      labelSaveLocation->setText(QString(filesystem::absolute(p).string().c_str()));
   }
}

void SaveMatchDialog::accept()
//----------------------------
{
   filesystem::path p(labelSaveLocation->text().toStdString());
   p = filesystem::absolute(p);
   std::string dir1 = dir.string();
   std::string dir2 = p.string();
   if ( (p == dir) || (dir2.size() <= dir1.size()) )
   {
      QMessageBox::warning(this, "Save Matches", "Directory name required - Cannot save to base directory.");
      editDirName->setFocus();
//      QDialog::reject();
      return;
   }

   std::string s = dir2.substr(0, dir1.size());
   if (s != dir1)
   {
      QMessageBox::warning(this, "Save Matches", "Directory name required - Cannot save to base directory.");
      editDirName->setFocus();
      return;
   }
   setResult(1);
   QDialog::accept();
}

ThreeDCoordsDialog::ThreeDCoordsDialog(cv::KeyPoint& kp, cv::Mat* desc,
                                       std::function<void(cv::KeyPoint&, cv::Mat&, cv::Point3d&)>& callback,
                                       QWidget* parent, double z_depth, double x, double y,
                                       bool isHorizontal) : QDialog(parent), callback(callback), x_text(new QLineEdit(this)),
                                                            y_text(new QLineEdit(this)), z_text(new QLineEdit(this)), keypoint(kp)
//-----------------------------------------------------------------------------------------------------------------------
{
   if (desc != nullptr)
      desc->copyTo(descriptor);
   QVBoxLayout* layout = new QVBoxLayout(this);
   QFormLayout* form_layout = new QFormLayout(this);
   x_text->setValidator(new QDoubleValidator); y_text->setValidator(new QDoubleValidator); z_text->setValidator(new QDoubleValidator);
   if ( (! std::isnan(z_depth)) && (z_depth > 0) )
      z_text->setText(QString::number(z_depth));
   if (! std::isnan(x))
      x_text->setText(QString::number(x));
   if (! std::isnan(y))
      y_text->setText(QString::number(y));
   if (isHorizontal)
   {
      QHBoxLayout* boxLayout = new QHBoxLayout(this);
      boxLayout->addWidget(x_text);
      boxLayout->addWidget(new QLabel(", "));
      boxLayout->addWidget(y_text);
      boxLayout->addWidget(new QLabel(", "));
      boxLayout->addWidget(z_text);
      form_layout->addRow("X, Y, Z: ", boxLayout);
   }
   else
   {
      form_layout->addRow("X:", x_text);
      form_layout->addRow("Y:", y_text);
      form_layout->addRow("Z:", z_text);
   }
   layout->addLayout(form_layout);
   QDialogButtonBox* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
   layout->addWidget(buttons);
   setLayout(layout);
   connect(buttons, SIGNAL(accepted()), this, SLOT(accept()));
   connect(buttons, SIGNAL(rejected()), this, SLOT(reject()));
   setWindowTitle("3D Coordinates");
   xw = yw = zw = std::numeric_limits<double>::quiet_NaN();
}

void ThreeDCoordsDialog::accept()
//-------------------------------
{
   bool is_ok;
   xw = x_text->text().toDouble(&is_ok);
   if (is_ok)
   {
      yw = y_text->text().toDouble(&is_ok);
      if (is_ok)
      {
         zw = z_text->text().toDouble(&is_ok);
         if ( (is_ok) && (callback) )
         {
            cv::Point3d wpt(xw, yw, zw);
            callback(keypoint, descriptor, wpt);
         }
         QDialog::accept();
         return;
      }
   }
   xw = yw = zw = std::numeric_limits<double>::quiet_NaN();
   std::stringstream ss;
   ss << "Invalid value (" << x_text->text().toStdString() << ", " << y_text->text().toStdString() << ", "
      << z_text->text().toStdString() << "). Point rejected";
   QMessageBox::warning(this, "Save 3D Coordinates", ss.str().c_str());
//   QDialog::accept();
}