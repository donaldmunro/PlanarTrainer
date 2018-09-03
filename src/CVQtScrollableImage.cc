#include <iostream>

#include "CVQtScrollableImage.h"

#include <QScrollBar>
#include <QMouseEvent>

CVQtScrollableImage::CVQtScrollableImage(QWidget *parent) : QWidget(parent), imageLabel(this), scrollArea(this)
//-----------------------------------------------------------------------------------------------------
{
//   setAttribute(Qt::WA_StaticContents);
   setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);

   QImage qimg(1024, 768, QImage::Format_RGB888);
   qimg.fill(qRgba(0, 0, 0, 0));
   imageLabel.setBackgroundRole(QPalette::Base);
//   imageLabel.setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
   QPixmap pixels = QPixmap::fromImage(qimg);
   imageLabel.setPixmap(pixels);
   imageLabel.adjustSize();
   //imageLabel.setScaledContents(true);

   scrollArea.setBackgroundRole(QPalette::Dark);
   scrollArea.setWidget(&imageLabel);
   scrollArea.setWidgetResizable(true);
   scrollArea.setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
//   scrollArea->setVisible(false);
   //  setCentralWidget(scrollArea.get());
}


void CVQtScrollableImage::set_image(cv::Mat& img, bool isBGR, bool isMono)
//-----------------------------------------------------------
{
   if (img.empty())
   {
      image.release();
      QImage qimg = QPixmap(1, 1).toImage();
      QPixmap pixels = QPixmap::fromImage(qimg);
      imageLabel.setPixmap(pixels);
      imageLabel.adjustSize();
   }
   else
   {
      if (isBGR)
         cv::cvtColor(img, image, CV_BGR2RGB);
      else
         img.copyTo(image);
      QImage::Format format = (isMono) ? QImage::Format_Mono : QImage::Format_RGB888;
      QImage qimg(image.data, image.cols, image.rows, format);
      QSize sze = this->parentWidget()->size();
      QPixmap pixels = QPixmap::fromImage(qimg);
      imageLabel.setPixmap(pixels);
      imageLabel.adjustSize();
      scrollArea.setVisible(true);
      scrollArea.setFixedSize(sze);
      scrollArea.setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
      scrollArea.setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
      update();
      updateGeometry();
//   std::cout << "CVQtScrollableImage::set_image " << scrollArea.size().width() << "x" << scrollArea.size().height() << " "
//             << imageLabel.size().width() << "x" << imageLabel.size().height() << " "
//             << pixels.size().width() << "x" << pixels.size().height() << " "
//             << image.cols << "x" << image.rows << std::endl;
   }
}

void CVQtScrollableImage::mousePressEvent(QMouseEvent *event)
//-------------------------------------------------------
{
   QWidget::mousePressEvent(event);
   if (on_roi_selection)
      drag_start = event->pos();
   if ( (on_left_click) || (on_right_click) )
   {
      QScrollBar *hbar = scrollArea.horizontalScrollBar();
      int x = hbar->value();
      QScrollBar *vbar = scrollArea.verticalScrollBar();
      int y = vbar->value();
      bool is_shift = static_cast<bool>(event->modifiers() & Qt::ShiftModifier);
      bool is_ctrl = static_cast<bool>(event->modifiers() & Qt::ControlModifier);
      bool is_alt = static_cast<bool>(event->modifiers() & Qt::AltModifier);
      cv::Point2f pt(drag_start.x() + x, drag_start.y() + y);
      if(event->button() == Qt::RightButton)
         on_right_click(pt, is_shift, is_ctrl, is_alt);
      else
         on_left_click(pt, is_shift, is_ctrl, is_alt);
   }

//   drag_show.reset(new QRubberBand(QRubberBand::Rectangle, this));
//   drag_show->setGeometry(QRect(drag_start, QSize()));
//   drag_show->show();
}

void CVQtScrollableImage::mouseMoveEvent(QMouseEvent *event)
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

void CVQtScrollableImage::mouseReleaseEvent(QMouseEvent *event)
//---------------------------------------------------------
{
   QWidget::mouseReleaseEvent(event);
   if ( (on_roi_selection) && (drag_show) )
   {
      drag_show->hide();
      drag_show.reset(nullptr);
      QScrollBar *hbar = scrollArea.horizontalScrollBar();
      int x = hbar->value();
      QScrollBar *vbar = scrollArea.verticalScrollBar();
      int y = vbar->value();
      if ( (drag_rect.width() > 2) && (drag_rect.height() > 2))
      {
         cv::Rect roi;
         roi.x = x + drag_rect.left();
         roi.y = y + drag_rect.top();
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

void CVQtScrollableImage::resizeEvent(QResizeEvent *event)
//-----------------------------------------------------
{
   QWidget::resizeEvent(event);
   QSize sze = this->parentWidget()->size();
   scrollArea.setFixedSize(sze);
}