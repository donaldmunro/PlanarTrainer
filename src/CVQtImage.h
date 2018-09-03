#ifndef _CVQTIMAGE_H_
#define _CVQTIMAGE_H_

#include <memory>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <QWidget>
#include <QLabel>
#include <QtWidgets/QRubberBand>

class CVQtImage : public QWidget
//====================================
{
   Q_OBJECT
//   Q_PROPERTY(cv::Mat image READ get_image WRITE set_image)

public:
   explicit CVQtImage(QWidget* parent = nullptr, int initialWidth =1, int initialHeight =1);

   ~CVQtImage() override
   //-----------------------------
   {
      //scrollArea.setWidget(nullptr);
      if (! image.empty())
         image.release();
   }

   void set_image(cv::Mat& img, bool isBGR =true, bool isMono =false);
   const cv::Mat& get_image() const { return image; }
   void set_roi_callback(std::function<void(cv::Mat&, cv::Rect&, bool, bool, bool)>& callback) { on_roi_selection = callback; }
   void set_mouse_click_callback(std::function<void(cv::Point2f&, bool, bool, bool)>& callback) { on_mouse_clicked = callback; }

   //QSize sizeHint() const override { return ((! display.empty()) ? QSize(display.cols, display.rows) : QSize(1024, 768)); }

protected:
   void mousePressEvent(QMouseEvent *event) override;
   void mouseMoveEvent(QMouseEvent *event) override;
   void mouseReleaseEvent(QMouseEvent *event) override;
   void resizeEvent(QResizeEvent *event) override;
//   void  keyPressEvent(QKeyEvent *event) override;

private:
   QLabel imageLabel;
   int initial_width, initial_height;
   cv::Mat image, display;
   QPoint drag_start;
   QRect drag_rect;
   bool is_mono = false, is_resizing = false;
   std::unique_ptr<QRubberBand> drag_show;
   std::function<void(cv::Point2f&, bool, bool, bool)> on_mouse_clicked;
   std::function<void(cv::Mat&, cv::Rect&, bool, bool, bool)> on_roi_selection;
   std::unique_ptr<QTimer> resize_timer_ptr;

   void resize();

private slots:
   void resize_timeout();
};
#endif //PNPTRAINER_CVQTSCROLLABLEIMAGE_H
