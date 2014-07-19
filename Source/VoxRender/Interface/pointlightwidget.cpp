/* ===========================================================================

	Project: VoxRender

	Description: Provides a control interface for point light sources

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
#include "ui_pointlightwidget.h"
#include "pointlightwidget.h"
#include "mainwindow.h"

// Include Dependencies
#include "utilities.h"

// VoxLib Dependencies
#include "VoxScene/Light.h"
#include "VoxLib/Core/format.h"
#include "VoxLib/Action/ActionManager.h"

// QT Includes
#include <QtWidgets/QMessageBox>

using namespace vox;

// --------------------------------------------------------------------
//  Constructor - Initialize the widget ui
// --------------------------------------------------------------------
PointLightWidget::PointLightWidget(QWidget * parent, void * userInfo, std::shared_ptr<Light> light) : 
    QWidget(parent), 
    ui(new Ui::PointLightWidget),
    m_userInfo(userInfo),
    m_light(light),
    m_colorButton(new QColorPushButton()),
    m_ignore(false)
{
	ui->setupUi(this);

    ui->layout_colorButton->addWidget(m_colorButton);
    connect(m_colorButton, SIGNAL(currentColorChanged(const QColor&)), this, SLOT(colorChanged(const QColor&)));

    sceneChanged();
}

// --------------------------------------------------------------------
//  Destructor
// --------------------------------------------------------------------
PointLightWidget::~PointLightWidget()
{
    delete m_colorButton;
    delete ui;
}

// --------------------------------------------------------------------
//  Synchronizes the widget with the light
// --------------------------------------------------------------------
void PointLightWidget::sceneChanged()
{
    m_ignore = true;
    
    // Synchronize the widget controls with the associated light
    auto position = m_light->position();
    auto distance = position.length();
    float phi     = acos(position[1] / distance) / M_PI * 180.0f;
    float theta   = atan2(position[2], position[0])  / M_PI * 180.0f;
    float intensity = m_light->color().fold(high);

    ui->doubleSpinBox_distance->setValue(distance);
    ui->doubleSpinBox_latitude->setValue(phi);
    ui->doubleSpinBox_longitude->setValue(theta);
    ui->doubleSpinBox_intensity->setValue(intensity);
    
    // Update the color button
    Vector3f color = intensity ? m_light->color() / intensity * 255.0f : Vector3f(0.0f, 0.0f, 0.0f);
    m_colorButton->setColor(QColor(color[0], color[1], color[2]), true); 

    m_ignore = false;
}

// --------------------------------------------------------------------
//  Synchronizes the light widget's position controls with the scene
// --------------------------------------------------------------------
void PointLightWidget::update()
{
    if (m_ignore) return;

    auto lock = MainWindow::instance->scene()->lock(m_userInfo);

    auto latitude  = ui->doubleSpinBox_latitude->value() / 180.0 * M_PI;
    auto longitude = ui->doubleSpinBox_longitude->value() / 180.0 * M_PI;
    auto distance  = ui->doubleSpinBox_distance->value();

    float sl = sin(latitude);

    m_light->setPositionX(sl * cos(longitude) * distance);
    m_light->setPositionY(cos(latitude)       * distance);
    m_light->setPositionZ(sl * sin(longitude) * distance);

    QColor color = m_colorButton->getColor();
    float  scale = ui->doubleSpinBox_intensity->value() / 255.0f;
    m_light->setColor(Vector3f(color.red(), color.green(), color.blue()) * scale);

    m_light->setDirty();
}

// --------------------------------------------------------------------
//  Toggles the light's visibility depending on the display status
// --------------------------------------------------------------------
void PointLightWidget::changeEvent(QEvent * event)
{
    if (event->type() != QEvent::EnabledChange ||
        isEnabled() == m_light->isVisible()) return;
    
    auto lock = MainWindow::instance->scene()->lock(m_userInfo);

    m_light->setVisible(isEnabled());
    m_light->setDirty(); 
}

// --------------------------------------------------------------------
//  Signals a color change in the color selection widget
// --------------------------------------------------------------------
void PointLightWidget::colorChanged(QColor const& color)
{
    update();
}

// --------------------------------------------------------------------
//  Spinbox <-> Slider connections
// --------------------------------------------------------------------
void PointLightWidget::on_horizontalSlider_intensity_valueChanged(int value)
{
    Utilities::forceSbToSl(
        ui->doubleSpinBox_intensity,
        ui->horizontalSlider_intensity,
        value);
    
    update();
}
void PointLightWidget::on_doubleSpinBox_intensity_valueChanged(double value)
{
    Utilities::forceSlToSb(
        ui->horizontalSlider_intensity,
        ui->doubleSpinBox_intensity,
        value);
    
    update();
}
void PointLightWidget::on_horizontalSlider_latitude_valueChanged(int value)
{
    Utilities::forceSbToSl(
        ui->doubleSpinBox_latitude,
        ui->horizontalSlider_latitude,
        value);
    
    update();
}
void PointLightWidget::on_doubleSpinBox_latitude_valueChanged(double value)
{
    Utilities::forceSlToSb(
        ui->horizontalSlider_latitude,
        ui->doubleSpinBox_latitude,
        value);
    
    update();
}
void PointLightWidget::on_horizontalSlider_longitude_valueChanged(int value)
{
    Utilities::forceSbToSl(
        ui->doubleSpinBox_longitude,
        ui->horizontalSlider_longitude,
        value);
    
    update();
}
void PointLightWidget::on_doubleSpinBox_longitude_valueChanged(double value)
{
    Utilities::forceSlToSb(
        ui->horizontalSlider_longitude,
        ui->doubleSpinBox_longitude,
        value);
    
    update();
}
void PointLightWidget::on_horizontalSlider_distance_valueChanged(int value)
{
    Utilities::forceSbToSl(
        ui->doubleSpinBox_distance,
        ui->horizontalSlider_distance,
        value);
    
    update();
}
void PointLightWidget::on_doubleSpinBox_distance_valueChanged(double value)
{
    Utilities::forceSlToSb(
        ui->horizontalSlider_distance,
        ui->doubleSpinBox_distance,
        value);
  
    update();
}