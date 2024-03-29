/* ===========================================================================

	Project: VoxRender

	Description: Implements the interface for point light source settings

    Copyright (C) 2012-2013 Lucas Sherman

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
#include "ui_clipplanewidget.h"
#include "clipplanewidget.h"

// Include Dependencies
#include "mainwindow.h"
#include "utilities.h"

// VoxLib Dependencies
#include "VoxLib/Core/format.h"
#include "VoxScene/Primitive.h"
#include "VoxScene/PrimGroup.h"
#include "VoxLib/Action/ActionManager.h"

// QT Includes
#include <QtWidgets/QMessageBox>

using namespace vox;

// --------------------------------------------------------------------
//  Constructor - Initialize the widget ui
// --------------------------------------------------------------------
ClipPlaneWidget::ClipPlaneWidget(QWidget * parent, void * userInfo, std::shared_ptr<Plane> plane) : 
    QWidget(parent), 
    ui(new Ui::ClipPlaneWidget),
    m_plane(plane),
    m_userInfo(userInfo),
    m_ignore(false)
{
	ui->setupUi(this);

    sceneChanged();
}

// --------------------------------------------------------------------
//  Destructor
// --------------------------------------------------------------------
ClipPlaneWidget::~ClipPlaneWidget()
{
    delete ui;
}

// --------------------------------------------------------------------
//  Synchronizes the widget's controls with the scene
// --------------------------------------------------------------------
void ClipPlaneWidget::sceneChanged()
{
    m_ignore = true;
    
    auto normal = m_plane->normal();
    auto pos    = normal * m_plane->distance();
    ui->doubleSpinBox_x->setValue(pos[0]);
    ui->doubleSpinBox_y->setValue(pos[1]);
    ui->doubleSpinBox_z->setValue(pos[2]);

    float phi      = acos(normal[1]) / M_PI * 180.0f;
    float theta    = atan2(normal[2], normal[0])  / M_PI * 180.0f;

    ui->doubleSpinBox_pitch->setValue(phi);
    ui->doubleSpinBox_yaw->setValue(theta);

    m_ignore = false;
}

// --------------------------------------------------------------------
//  Synchronizes the scene with the widget's controls
// --------------------------------------------------------------------
void ClipPlaneWidget::update()
{
    if (m_ignore) return;

    // Compute the new normal vector of the plane
    double pitch  = ui->doubleSpinBox_pitch->value() / 180.0 * M_PI;
    double yaw    = ui->doubleSpinBox_yaw->value()   / 180.0 * M_PI;
    float  sp     = sin(pitch);

    Vector3f normal = Vector3f(
        sp * cos(yaw),
        cos(pitch),
        sp * sin(yaw)).normalized();
    m_plane->setNormal(normal);

    // Compute the minimum distance from the origin
    Vector3f position(
        ui->doubleSpinBox_x->value(),
        ui->doubleSpinBox_y->value(),
        ui->doubleSpinBox_z->value());
    float distance = normal.dot(position);
    m_plane->setDistance(distance);
    
    m_plane->setDirty();
}

// --------------------------------------------------------------------
//  Toggles the planes visibility depending on the display status
// --------------------------------------------------------------------
void ClipPlaneWidget::changeEvent(QEvent * event)
{
    if (event->type() != QEvent::EnabledChange ||
        isEnabled() == m_plane->isVisible()) return;
    
    auto lock = MainWindow::instance->scene()->lock(m_userInfo);

    m_plane->setVisible(isEnabled());
    m_plane->setDirty(); 
}

// --------------------------------------------------------------------
//  Synchronizes the slider and spinbox elements
// --------------------------------------------------------------------
void ClipPlaneWidget::on_horizontalSlider_pitch_valueChanged(int value)
{
    Utilities::forceSbToSl(
        ui->doubleSpinBox_pitch,
        ui->horizontalSlider_pitch,
        value);
    
    update();
}
void ClipPlaneWidget::on_horizontalSlider_yaw_valueChanged(int value)
{
    Utilities::forceSbToSl(
        ui->doubleSpinBox_yaw,
        ui->horizontalSlider_yaw,
        value);
    
    update();
}
void ClipPlaneWidget::on_horizontalSlider_x_valueChanged(int value)
{
    Utilities::forceSbToSl(
        ui->doubleSpinBox_x,
        ui->horizontalSlider_x,
        value);
    
    update();
}
void ClipPlaneWidget::on_horizontalSlider_y_valueChanged(int value)
{
    Utilities::forceSbToSl(
        ui->doubleSpinBox_y,
        ui->horizontalSlider_y,
        value);
    
    update();
}
void ClipPlaneWidget::on_horizontalSlider_z_valueChanged(int value)
{
    Utilities::forceSbToSl(
        ui->doubleSpinBox_z,
        ui->horizontalSlider_z,
        value);
    
    update();
}
void ClipPlaneWidget::on_doubleSpinBox_pitch_valueChanged(double value)
{
    Utilities::forceSlToSb(
        ui->horizontalSlider_pitch,
        ui->doubleSpinBox_pitch,
        value);
    
    update();
}
void ClipPlaneWidget::on_doubleSpinBox_yaw_valueChanged(double value)
{
    Utilities::forceSlToSb(
        ui->horizontalSlider_yaw,
        ui->doubleSpinBox_yaw,
        value);
    
    update();
}
void ClipPlaneWidget::on_doubleSpinBox_x_valueChanged(double value)
{
    Utilities::forceSlToSb(
        ui->horizontalSlider_x,
        ui->doubleSpinBox_x,
        value);
    
    update();
}
void ClipPlaneWidget::on_doubleSpinBox_y_valueChanged(double value)
{
    Utilities::forceSlToSb(
        ui->horizontalSlider_y,
        ui->doubleSpinBox_y,
        value);
    
    update();
}
void ClipPlaneWidget::on_doubleSpinBox_z_valueChanged(double value)
{
    Utilities::forceSlToSb(
        ui->horizontalSlider_z,
        ui->doubleSpinBox_z,
        value);
    
    update();
}
