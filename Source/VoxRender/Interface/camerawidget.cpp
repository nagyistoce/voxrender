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

// Include Headers
#include "camerawidget.h"
#include "ui_camerawidget.h"

// Include Dependencies
#include "mainwindow.h"
#include "utilities.h"

// --------------------------------------------------------------------
//  Constructor
// --------------------------------------------------------------------
CameraWidget::CameraWidget(QWidget *parent) :
    QWidget(parent), 
    ui(new Ui::CameraWidget),
    m_dirty(false),
    m_filmDirty(false)
{
    ui->setupUi(this);
}
    
// --------------------------------------------------------------------
//  Destructor
// --------------------------------------------------------------------
CameraWidget::~CameraWidget()
{
    delete ui;
}

// --------------------------------------------------------------------
//  Synchronizes the widget controls with the current scene 
// --------------------------------------------------------------------
void CameraWidget::synchronizeView()
{
    // Synchronize the camera object controls
    vox::Camera & camera = *MainWindow::instance->scene().camera;
    
    float fov = camera.fieldOfView() * 180.0f / M_PI;
    ui->doubleSpinBox_camFov->setValue( fov );
    
    ui->doubleSpinBox_focal->setValue( camera.focalDistance() );
    ui->doubleSpinBox_aperture->setValue( camera.apertureSize() );

    // Synchronize the film object controls
    vox::Film & film = *MainWindow::instance->scene().film;

    ui->spinBox_filmWidth->setValue( film.width() );
    ui->spinBox_filmHeight->setValue( film.height() );

    // Clean the view to prevent forced update
    m_filmDirty = false; m_dirty = false;
}

// --------------------------------------------------------------------
//  Applies widget control changes to the scene camera 
// --------------------------------------------------------------------
void CameraWidget::processInteractions()
{
    if (m_dirty)
    {
        m_dirty = false;

        vox::Camera & camera = *MainWindow::instance->scene().camera;
        
        float fov = ui->doubleSpinBox_camFov->value() / 180.0f * M_PI;
        camera.setFieldOfView( fov );

        camera.setFocalDistance( ui->doubleSpinBox_focal->value() );
        camera.setApertureSize( ui->doubleSpinBox_aperture->value() );
    }

    if (m_filmDirty)
    {
        m_filmDirty = false;

        vox::Film & film = *MainWindow::instance->scene().film;

        film.setWidth( ui->spinBox_filmWidth->value() );
        film.setHeight( ui->spinBox_filmHeight->value() );
    }
}

// --------------------------------------------------------------------
//  Update the scene film height to match the newly specified values
// --------------------------------------------------------------------
void CameraWidget::on_spinBox_filmHeight_valueChanged(int value)
{
    m_filmDirty = true;
}

// --------------------------------------------------------------------
//  Update the scene film width to match the newly specified values
// --------------------------------------------------------------------
void CameraWidget::on_spinBox_filmWidth_valueChanged(int value)
{
    m_filmDirty = true;
}

// --------------------------------------------------------------------
//  Modify the associated double spinbox to reflect slide value change
// --------------------------------------------------------------------
void CameraWidget::on_horizontalSlider_camFov_valueChanged(int value)
{
    Utilities::forceSbToSl(
        ui->doubleSpinBox_camFov,
        ui->horizontalSlider_camFov,
        value);
    
    m_dirty = true;
}

// --------------------------------------------------------------------
//  Modify the associated double spinbox to reflect slide value change
// --------------------------------------------------------------------
void CameraWidget::on_horizontalSlider_aperture_valueChanged(int value)
{
    Utilities::forceSbToSl(
        ui->doubleSpinBox_aperture,
        ui->horizontalSlider_aperture,
        value);
    
    m_dirty = true;
}

// --------------------------------------------------------------------
//  Modify the associated double spinbox to reflect slide value change
// --------------------------------------------------------------------
void CameraWidget::on_horizontalSlider_focal_valueChanged(int value)
{
    Utilities::forceSbToSl(
        ui->doubleSpinBox_focal,
        ui->horizontalSlider_focal,
        value);
    
    m_dirty = true;
}

// --------------------------------------------------------------------
//  Modify the associated slider to reflect spinBox value change
// --------------------------------------------------------------------
void CameraWidget::on_doubleSpinBox_camFov_valueChanged(double value)
{
    Utilities::forceSlToSb(
        ui->horizontalSlider_camFov,
        ui->doubleSpinBox_camFov,
        value);
    
    m_dirty = true;
}

// --------------------------------------------------------------------
//  Modify the associated slider to reflect spinBox value change
// --------------------------------------------------------------------
void CameraWidget::on_doubleSpinBox_aperture_valueChanged(double value)
{
    Utilities::forceSlToSb(
        ui->horizontalSlider_aperture,
        ui->doubleSpinBox_aperture,
        value);
    
    m_dirty = true;
}

// --------------------------------------------------------------------
//  Modify the associated slider to reflect spinBox value change
// --------------------------------------------------------------------
void CameraWidget::on_doubleSpinBox_focal_valueChanged(double value)
{
    Utilities::forceSlToSb(
        ui->horizontalSlider_focal,
        ui->doubleSpinBox_focal,
        value);

    m_dirty = true;
}