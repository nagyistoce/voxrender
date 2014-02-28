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

// QT Includes
#include <QtWidgets/QMessageBox>

using namespace vox;

// --------------------------------------------------------------------
//  Constructor - Initialize the widget ui
// --------------------------------------------------------------------
PointLightWidget::PointLightWidget(QWidget * parent, std::shared_ptr<Light> light) : 
    QWidget(parent), 
    ui(new Ui::PointLightWidget),
    m_light(light),
    m_colorButton(new QColorPushButton()),
    m_ignore(true)
{
	ui->setupUi(this);

    // Synchronize the widget controls with the associated light
    float distance  = light->position().length();
    float partial   = sqrt( pow(light->positionX(), 2) + pow(light->positionZ(), 2) );
    float phi       = asin(light->positionY() / distance) / M_PI * 180.0f;
    float theta     = - acos(light->positionX() / partial)  / M_PI * 180.0f;
    float intensity = light->color().fold(high);
    if (light->positionX() < 0) theta = - theta;

    ui->doubleSpinBox_distance->setValue(distance);
    ui->doubleSpinBox_latitude->setValue(phi);
    ui->doubleSpinBox_longitude->setValue(theta);
    ui->doubleSpinBox_intensity->setValue(intensity);

    // Initialize the color control elements
    m_colorButton = new QColorPushButton();
    Vector3f color = intensity ? light->color() / intensity * 255.0f : Vector3f(0.0f, 0.0f, 0.0f);
    m_colorButton->setColor(QColor(color[0], color[1], color[2]), true); 
    ui->layout_colorButton->addWidget(m_colorButton);
    connect(m_colorButton, SIGNAL(currentColorChanged(const QColor&)), this, SLOT(colorChanged(const QColor&)));

    m_ignore = false;
}

// --------------------------------------------------------------------
//  Destructor
// --------------------------------------------------------------------
PointLightWidget::~PointLightWidget()
{
    auto & scene = MainWindow::instance->scene();
    if (scene.lightSet) 
    {
        scene.lightSet->lock();
        scene.lightSet->remove(m_light);
        scene.lightSet->setDirty();
        scene.lightSet->unlock();
    }

    delete m_colorButton;
    delete ui;
}

// --------------------------------------------------------------------
//  Synchronizes the light widget's position controls with the scene
// --------------------------------------------------------------------
void PointLightWidget::update()
{
    if (m_ignore) return;

    m_light->lock();

        double latitude  = ui->doubleSpinBox_latitude->value() / 180.0 * M_PI;
        double longitude = ui->doubleSpinBox_longitude->value() / 180.0 * M_PI;
        double distance  = ui->doubleSpinBox_distance->value();

        float cl = cos(latitude);

        m_light->setPositionX(cl * cos(longitude) * distance);
        m_light->setPositionY(sin(latitude)       * distance);
        m_light->setPositionZ(cl * sin(longitude) * distance);

        QColor color = m_colorButton->getColor();
        float  scale = ui->doubleSpinBox_intensity->value() / 255.0f;
        m_light->setColor(Vector3f(color.red(), color.green(), color.blue()) * scale);

        m_light->setDirty();

    m_light->unlock();
}

// --------------------------------------------------------------------
//  Toggles the light's visibility depending on the display status
// --------------------------------------------------------------------
void PointLightWidget::changeEvent(QEvent * event)
{
    if (event->type() == QEvent::EnabledChange)
    {
        auto scene = MainWindow::instance->scene();
        if (!scene.lightSet) return;
        
        if (MainWindow::instance->renderState() != RenderState_Rendering) return;

        // :TODO: Add isVisible member to scene classes
        scene.lightSet->lock();
            if (!isEnabled()) scene.lightSet->remove(m_light);
            else 
            {
                auto children = scene.lightSet->lights();
                if (std::find(children.begin(), children.end(), m_light) == children.end())
                    scene.lightSet->add(m_light);
            }
        scene.lightSet->setDirty();
        scene.lightSet->unlock();
    }
}

// --------------------------------------------------------------------
//  Modify the associated double spinbox to reflect slide value change
// --------------------------------------------------------------------
void PointLightWidget::on_horizontalSlider_intensity_valueChanged(int value)
{
    Utilities::forceSbToSl(
        ui->doubleSpinBox_intensity,
        ui->horizontalSlider_intensity,
        value);
    
    update();
}

// --------------------------------------------------------------------
//  Modifies the intensity component of the light's emissions
// --------------------------------------------------------------------
void PointLightWidget::on_doubleSpinBox_intensity_valueChanged(double value)
{
    Utilities::forceSlToSb(
        ui->horizontalSlider_intensity,
        ui->doubleSpinBox_intensity,
        value);
    
    update();
}

// --------------------------------------------------------------------
//  Modify the associated double spinbox to reflect slide value change
// --------------------------------------------------------------------
void PointLightWidget::on_horizontalSlider_latitude_valueChanged(int value)
{
    Utilities::forceSbToSl(
        ui->doubleSpinBox_latitude,
        ui->horizontalSlider_latitude,
        value);
    
    update();
}

// --------------------------------------------------------------------
//  Modifies the latitude of the light's position relative to the data
// --------------------------------------------------------------------
void PointLightWidget::on_doubleSpinBox_latitude_valueChanged(double value)
{
    Utilities::forceSlToSb(
        ui->horizontalSlider_latitude,
        ui->doubleSpinBox_latitude,
        value);
    
    update();
}

// --------------------------------------------------------------------
//  Modify the associated double spinbox to reflect slide value change
// --------------------------------------------------------------------
void PointLightWidget::on_horizontalSlider_longitude_valueChanged(int value)
{
    Utilities::forceSbToSl(
        ui->doubleSpinBox_longitude,
        ui->horizontalSlider_longitude,
        value);
    
    update();
}

// --------------------------------------------------------------------
//  Modifies the longitude of the light's position relative to the data
// --------------------------------------------------------------------
void PointLightWidget::on_doubleSpinBox_longitude_valueChanged(double value)
{
    Utilities::forceSlToSb(
        ui->horizontalSlider_longitude,
        ui->doubleSpinBox_longitude,
        value);
    
    update();
}

// --------------------------------------------------------------------
//  Modify the associated double spinbox to reflect slide value change
// --------------------------------------------------------------------
void PointLightWidget::on_horizontalSlider_distance_valueChanged(int value)
{
    Utilities::forceSbToSl(
        ui->doubleSpinBox_distance,
        ui->horizontalSlider_distance,
        value);
    
    update();
}

// --------------------------------------------------------------------
//  Modifies the distance of the light from the data
// --------------------------------------------------------------------
void PointLightWidget::on_doubleSpinBox_distance_valueChanged(double value)
{
    Utilities::forceSlToSb(
        ui->horizontalSlider_distance,
        ui->doubleSpinBox_distance,
        value);
  
    update();
}

// --------------------------------------------------------------------
//  Signals a color change in the color selection widget
// --------------------------------------------------------------------
void PointLightWidget::colorChanged(QColor const& color)
{
    update();
}