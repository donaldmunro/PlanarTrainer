#include <iostream>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "CVQtImage.h"

#include <QScrollBar>
#include <QTimer>
#include <QMouseEvent>
#include <QImageReader>

CVQtImage::CVQtImage(QWidget *parent, int initialWidth, int initialHeight) : QWidget(parent), imageLabel(this),
                                                                             initial_width(initialWidth),
                                                                             initial_height(initialHeight)
//-----------------------------------------------------------------------------------------------------
{
   setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
   QImage qimg(initialWidth, initialHeight, QImage::Format_RGB888);
   qimg.fill(qRgba(0, 0, 0, 0));
   imageLabel.setStyleSheet("margin-left: 1px; border-radius: 2px");
   imageLabel.setBackgroundRole(QPalette::Base);
   QPixmap pixels = QPixmap::fromImage(qimg);
   imageLabel.setPixmap(pixels);
   imageLabel.adjustSize();
   imageLabel.setScaledContents(true);
}


void CVQtImage::set_image(cv::Mat& img, bool isBGR, bool isMono)
//-----------------------------------------------------------
{
   if (img.empty())
   {
      image.release();
      QImage qimg(initial_width, initial_height, QImage::Format_RGB888);
      qimg.fill(qRgba(0, 0, 0, 0));
      QPixmap pixels = QPixmap::fromImage(qimg);
      imageLabel.setPixmap(pixels);
      imageLabel.adjustSize();
   }
   else
   {
      if (! isBGR)
         cv::cvtColor(img, image, CV_RGB2BGR);
      else
         img.copyTo(image);

      QSize sze = this->parentWidget()->size();
      int h = sze.height(), w = sze.width();
      cv::resize(image, img, cv::Size(w, h), 0, 0, cv::INTER_LANCZOS4);
      cv::cvtColor(img, display, CV_BGR2RGB);
      QImage::Format format = (isMono) ? QImage::Format_Mono : QImage::Format_RGB888;
      QImage qimg(display.data, display.cols, display.rows, format);
      QPixmap pixels = QPixmap::fromImage(qimg);
      imageLabel.setPixmap(pixels);
      imageLabel.adjustSize();
      updateGeometry();
   }
   is_mono = isMono;
}

void CVQtImage::mousePressEvent(QMouseEvent *event)
//-------------------------------------------------------
{
   QWidget::mousePressEvent(event);
   if (on_roi_selection)
      drag_start = event->pos();
   if (on_mouse_clicked)
   {
      bool is_shift = static_cast<bool>(event->modifiers() & Qt::ShiftModifier);
      bool is_ctrl = static_cast<bool>(event->modifiers() & Qt::ControlModifier);
      bool is_alt = static_cast<bool>(event->modifiers() & Qt::AltModifier);
      cv::Point2f pt(drag_start.x(), drag_start.y());
      on_mouse_clicked(pt, is_shift, is_ctrl, is_alt);
   }
}

void CVQtImage::mouseMoveEvent(QMouseEvent *event)
//------------------------------------------------------
{
   QWidget::mouseMoveEvent(event);
   if (! on_roi_selection) return;
   drag_rect = QRect(drag_start, event->pos()).normalized();
   if (drag_rect.left() < 0) drag_rect.setLeft(0);
   if (drag_rect.top() < 0) drag_rect.setTop(0);
   if (drag_rect.right() > image.cols) drag_rect.setRight(image.cols - 1);
   if (drag_rect.bottom() > image.rows) drag_rect.setBottom(image.rows - 1);
   if ( (drag_rect.width() > 2) && (drag_rect.height() > 2) )
   {
      bool is_drag = (drag_show.get() != nullptr);
      if (! is_drag)
         drag_show.reset(new QRubberBand(QRubberBand::Rectangle, this));
      drag_show->setGeometry(drag_rect);
      if (! is_drag)
         drag_show->show();
   }
}

void CVQtImage::mouseReleaseEvent(QMouseEvent *event)
//---------------------------------------------------------
{
   QWidget::mouseReleaseEvent(event);
   if ( (on_roi_selection) && (drag_show) )
   {
      drag_show->hide();
      drag_show.reset(nullptr);
      if ( (drag_rect.width() > 2) && (drag_rect.height() > 2))
      {
         cv::Rect roi;
         roi.x = drag_rect.left();
         roi.y = drag_rect.top();
         roi.width = drag_rect.width();
         roi.height = drag_rect.height();
         if (roi.x < 0) roi.x = 0;
         if (roi.y < 0) roi.y = 0;
         if ((roi.x + roi.width) > image.cols)
            roi.width = image.cols - roi.x;
         if ((roi.y + roi.height) > image.rows)
            roi.height = image.rows - roi.y;
         cv::Mat m(image, roi);
         bool is_shift = static_cast<bool>(event->modifiers() & Qt::ShiftModifier);
         bool is_ctrl = static_cast<bool>(event->modifiers() & Qt::ControlModifier);
         bool is_alt = static_cast<bool>(event->modifiers() & Qt::AltModifier);
         on_roi_selection(m, roi, is_shift, is_ctrl, is_alt);
      }
   }
   drag_rect = QRect();
   drag_start = QPoint(-1, -1);
}

void CVQtImage::resizeEvent(QResizeEvent *event)
//-----------------------------------------------------
{
   QWidget::resizeEvent(event);
   if (is_resizing)
   {
      is_resizing = false;
      return;
   }
   if (! resize_timer_ptr)
   {
      resize_timer_ptr.reset(new QTimer(this));
      resize_timer_ptr->setSingleShot(true);
      resize_timer_ptr->connect(resize_timer_ptr.get(), SIGNAL(timeout()), this, SLOT(resize_timeout()));
   }
   if (resize_timer_ptr->isActive())
      resize_timer_ptr->stop();
   resize_timer_ptr->start(300);
}

void CVQtImage::resize_timeout()
//------------------------------
{
   is_resizing = true;
   if (resize_timer_ptr)
   {
      resize_timer_ptr->stop();
      resize_timer_ptr->disconnect(resize_timer_ptr.get(), SIGNAL(timeout()), nullptr, nullptr);
      resize_timer_ptr.reset(nullptr);
   }
   if (image.empty() || display.empty()) return;
   this->parentWidget()->updateGeometry();
   QSize sze = this->parentWidget()->size();
   int h = sze.height(), w = sze.width();
   imageLabel.resize(w, h);
}

//void CVQtImage::resize()
////----------------------
//{
//   if (image.empty()) return;
//   this->parentWidget()->updateGeometry();
//   QSize sze = this->parentWidget()->size();
//   int h = sze.height(), w = sze.width();
//   QRect geom = this->parentWidget()->geometry();
//   imageLabel.setGeometry(geom);
//
////   imageLabel.setFixedSize(sze);
//

//   imageLabel.resize(w, h);
//
////   if ( (display.empty()) || ( (display.rows != h) || (display.cols != w) ) )
////   {
////      cv::Mat img;
//////      if ( (image.rows > h) &&  (image.cols > w) )//shrink
//////         cv::resize(image, img, cv::Size(w, h), 0, 0, CV_INTER_AREA);
//////      else
//////         cv::resize(image, img, cv::Size(w, h), 0, 0, cv::INTER_LANCZOS4);
////      cv::resize(image, img, cv::Size(w, h), 0, 0, cv::INTER_LANCZOS4);
////      cv::cvtColor(img, display, CV_BGR2RGB);
////   }
////   cv::imwrite("image.png", image); cv::imwrite("display.png", display);
////   QImage::Format format = (is_mono) ? QImage::Format_Mono : QImage::Format_RGB888;
////   QImage qimg(display.data, display.cols, display.rows, format);
//////      QSize sze = this->parentWidget()->size();
////   QPixmap pixels = QPixmap::fromImage(qimg);
//////   QImageReader reader("display.png");
//////   pixels.fromImageReader(&reader);
////
//////   imageLabel.clear();
////   imageLabel.setGeometry(geom);
////   imageLabel.setPixmap(pixels);
//////   imageLabel.adjustSize();
////   imageLabel.update();
////   imageLabel.updateGeometry();
////   update();
////   updateGeometry();
////   show();
//}