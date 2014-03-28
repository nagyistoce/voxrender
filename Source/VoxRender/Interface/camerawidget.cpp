/* ===========================================================================

	Project: VoxRender - Camera Widget

	Description:
	 Implements a widget for controlling the scene camera settings

    Copyright (C) 2012-2014 Lucas Sherman

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
#include "VoxScene/Camera.h"
#include "Actions/CamEditAct.h"
#include "VoxLib/Action/ActionManager.h"

namespace {
namespace filescope {

    enum EditType
    {
        EditType_Film,
        EditType_Aperture,
        EditType_FieldOfView,
        EditType_Exposure,
    };

}
}

// --------------------------------------------------------------------
//  Constructor
// --------------------------------------------------------------------
CameraWidget::CameraWidget(QWidget *parent) :
    QWidget(parent), 
    ui(new Ui::CameraWidget),
    m_ignore(false)
{
    ui->setupUi(this);
    
    connect(MainWindow::instance, SIGNAL(sceneChanged()), this, SLOT(sceneChanged()));
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
void CameraWidget::sceneChanged()
{
    auto camera = MainWindow::instance->scene().camera;
    if (!camera) return;
    
    m_ignore = true;

    float fov = camera->fieldOfView() * 180.0f / M_PI;
    ui->doubleSpinBox_camFov->setValue( fov );
    
    ui->doubleSpinBox_focal->setValue( camera->focalDistance() );
    ui->doubleSpinBox_aperture->setValue( camera->apertureSize() );
    ui->spinBox_filmWidth->setValue( camera->filmWidth() );
    ui->spinBox_filmHeight->setValue( camera->filmHeight() );

    m_ignore = false;
}

// --------------------------------------------------------------------
//  Applies widget control changes to the scene camera 
// --------------------------------------------------------------------
void CameraWidget::updateCamera()
{
    if (m_ignore) return;
    auto camera = MainWindow::instance->scene().camera;
    if (!camera) return;

    camera->lock();

        float fov = ui->doubleSpinBox_camFov->value() / 180.0f * M_PI;
        camera->setFieldOfView( fov );

        camera->setFocalDistance( ui->doubleSpinBox_focal->value() );
        camera->setApertureSize( ui->doubleSpinBox_aperture->value() );
        camera->setDirty();

    camera->unlock();
}

// --------------------------------------------------------------------
//  Applies control changes the the scene camera's film settings
// --------------------------------------------------------------------
void CameraWidget::updateFilm()
{
    if (m_ignore) return;
    auto camera = MainWindow::instance->scene().camera;
    if (!camera) return;
    
    camera->lock();

        camera->setFilmWidth( ui->spinBox_filmWidth->value() );
        camera->setFilmHeight( ui->spinBox_filmHeight->value() );
        camera->setFilmDirty();

    camera->unlock();
}

// --------------------------------------------------------------------
//  Update the scene film height to match the newly specified values
// --------------------------------------------------------------------
void CameraWidget::on_spinBox_filmHeight_valueChanged(int value)
{
    updateFilm();
}

// --------------------------------------------------------------------
//  Update the scene film width to match the newly specified values
// --------------------------------------------------------------------
void CameraWidget::on_spinBox_filmWidth_valueChanged(int value)
{
    updateFilm();
}


// --------------------------------------------------------------------
//  Modify the associated double spinbox to reflect slide value change
// --------------------------------------------------------------------
void CameraWidget::on_horizontalSlider_exposure_valueChanged(int value)
{
    Utilities::forceSbToSl(
        ui->doubleSpinBox_exposure,
        ui->horizontalSlider_exposure,
        value);

    MainWindow::instance->m_renderer->setExposure(
        ui->doubleSpinBox_exposure->value());
}

// --------------------------------------------------------------------
//  Modify the associated slider to reflect spinBox value change
// --------------------------------------------------------------------
void CameraWidget::on_doubleSpinBox_exposure_valueChanged(double value)
{
    Utilities::forceSlToSb(
        ui->horizontalSlider_exposure,
        ui->doubleSpinBox_exposure,
        value);
    
    MainWindow::instance->m_renderer->setExposure(
        ui->doubleSpinBox_exposure->value());
}

// --------------------------------------------------------------------
//  Update the camera on control element changes
// --------------------------------------------------------------------
void CameraWidget::on_horizontalSlider_camFov_valueChanged(int value)
{
    Utilities::forceSbToSl(
        ui->doubleSpinBox_camFov,
        ui->horizontalSlider_camFov,
        value);
    
    updateCamera();
}
void CameraWidget::on_horizontalSlider_aperture_valueChanged(int value)
{
    Utilities::forceSbToSl(
        ui->doubleSpinBox_aperture,
        ui->horizontalSlider_aperture,
        value);
    
    updateCamera();
}
void CameraWidget::on_horizontalSlider_focal_valueChanged(int value)
{
    Utilities::forceSbToSl(
        ui->doubleSpinBox_focal,
        ui->horizontalSlider_focal,
        value);
    
    updateCamera();
}
void CameraWidget::on_doubleSpinBox_camFov_valueChanged(double value)
{
    Utilities::forceSlToSb(
        ui->horizontalSlider_camFov,
        ui->doubleSpinBox_camFov,
        value);
    
    updateCamera();
}
void CameraWidget::on_doubleSpinBox_aperture_valueChanged(double value)
{
    Utilities::forceSlToSb(
        ui->horizontalSlider_aperture,
        ui->doubleSpinBox_aperture,
        value);
    
    updateCamera();
}
void CameraWidget::on_doubleSpinBox_focal_valueChanged(double value)
{
    Utilities::forceSlToSb(
        ui->horizontalSlider_focal,
        ui->doubleSpinBox_focal,
        value);
    
    updateCamera();
}