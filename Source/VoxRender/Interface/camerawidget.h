/* ===========================================================================

	Project: VoxRender - Camera Widget

	Description:
	 Implements a widget for controlling the scene camera settings

    Copyright (C) 2012 Lucas Sherman

	Lucas Sherman, email: LucasASherman@gmail.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

=========================================================================== */

// Begin Definition
#ifndef CAMERA_WIDGET_H
#define CAMERA_WIDGET_H

// QT Includes
#include <QtWidgets/QWidget>

namespace Ui {
class CameraWidget;
}

// Camera Settings Widget
class CameraWidget : public QWidget
{
    Q_OBJECT
    
public:
    explicit CameraWidget(QWidget *parent = 0);

    ~CameraWidget( );

private:
    Ui::CameraWidget *ui;

    void updateCamera();
    void updateFilm();

    bool m_ignore;

private slots:
    void sceneChanged();

    void on_horizontalSlider_camFov_valueChanged(int value);
    void on_doubleSpinBox_camFov_valueChanged(double value);
    void on_horizontalSlider_aperture_valueChanged(int value);
    void on_doubleSpinBox_aperture_valueChanged(double value);
    void on_horizontalSlider_focal_valueChanged(int value);
    void on_doubleSpinBox_focal_valueChanged(double value);
    void on_horizontalSlider_exposure_valueChanged(int value);
    void on_doubleSpinBox_exposure_valueChanged(double value);

    void on_spinBox_filmHeight_valueChanged(int value);
    void on_spinBox_filmWidth_valueChanged(int value);
};

// End Definition
#endif // CAMERA_WIDGET_H
