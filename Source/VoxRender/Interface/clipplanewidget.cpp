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

// QT Includes
#include <QtWidgets/QMessageBox>

using namespace vox;

// File scope namespace
namespace {
namespace filescope {
}
}

// --------------------------------------------------------------------
//  Constructor - Initialize the widget ui
// --------------------------------------------------------------------
ClipPlaneWidget::ClipPlaneWidget(QWidget * parent, std::shared_ptr<Plane> plane) : 
    QWidget(parent), 
    ui(new Ui::ClipPlaneWidget),
    m_plane(plane),
    m_block(true)
{
	ui->setupUi(this);

    auto length = plane->normal().length();
    auto normal = plane->normal();
    auto pos    = normal * plane->distance();
    ui->doubleSpinBox_x->setValue(pos[0]);
    ui->doubleSpinBox_y->setValue(pos[1]);
    ui->doubleSpinBox_z->setValue(pos[2]);

    float partial  = sqrt(pow(normal[0], 2) + pow(normal[2], 2));
    float phi      = asin(normal[1]) / M_PI * 180.0f;
    float theta    = - acos(normal[0] / partial)  / M_PI * 180.0f;

    ui->doubleSpinBox_pitch->setValue(phi);
    ui->doubleSpinBox_yaw->setValue(theta);

    m_block = false;
}

// --------------------------------------------------------------------
//  Destructor
// --------------------------------------------------------------------
ClipPlaneWidget::~ClipPlaneWidget()
{
    auto cg = MainWindow::instance->scene().clipGeometry;
    
    if (cg) 
    {
        SceneLock lock(cg);
        cg->remove(m_plane);
        cg->setDirty();
    }

    delete ui;
}

// --------------------------------------------------------------------
//  Synchronizes the light widget's position controls with the scene
// --------------------------------------------------------------------
void ClipPlaneWidget::update()
{
    if (m_block) return;

    // Compute the new normal vector of the plane
    double latitude  = ui->doubleSpinBox_pitch->value() / 180.0 * M_PI;
    double longitude = ui->doubleSpinBox_yaw->value()   / 180.0 * M_PI;
    float  cl        = cos(latitude);

    Vector3f normal = Vector3f(
        cl * cos(longitude),
        sin(latitude),
        cl * sin(longitude)).normalized();
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
    if (event->type() == QEvent::EnabledChange)
    {
        auto scene = MainWindow::instance->scene();
        if (!scene.clipGeometry) return;

        if (!isEnabled()) scene.clipGeometry->remove(m_plane, true);
        else 
        {
            auto children = scene.clipGeometry->children();
            if (std::find(children.begin(), children.end(), m_plane) == children.end())
                scene.clipGeometry->add(m_plane, true);
        }
    }
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
