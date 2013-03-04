/* ===========================================================================

	Project: VoxRender - Point Light Interface

	Based on luxrender light group widget class.
	Lux Renderer website : http://www.luxrender.net 

	Description:
	 Implements the interface for point light source settings

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
#include "ui_pointlightwidget.h"
#include "pointlightwidget.h"

// Include Dependencies
#include "utilities.h"

// VoxLib Dependencies
#include "VoxLib/Scene/Light.h"

using namespace vox;

// --------------------------------------------------------------------
//  Constructor - Initialize the widget ui
// --------------------------------------------------------------------
PointLightWidget::PointLightWidget(QWidget * parent, std::shared_ptr<Light> light) : 
    QWidget(parent), 
    ui(new Ui::PointLightWidget),
    m_title("Point Light"),
    m_light(light)
{
	ui->setupUi(this);

    // Synchronize the widget controls with the associated light
    float distance = light->position().length();
    float partial  = sqrt( pow(light->positionX(), 2) + pow(light->positionZ(), 2) );
    float phi      = asin(light->positionY() / distance) / M_PI * 180.0f;
    float theta    = acos(light->positionX() / partial)  / M_PI * 180.0f;
    if (light->positionX() < 0) theta = - theta;

    ui->doubleSpinBox_distance->setValue( distance );
    ui->doubleSpinBox_latitude->setValue( phi );
    ui->doubleSpinBox_longitude->setValue( theta );

    m_dirty = false;
}

// --------------------------------------------------------------------
//  Destructor
// --------------------------------------------------------------------
PointLightWidget::~PointLightWidget()
{
    delete ui;
}

// --------------------------------------------------------------------
//  Synchronizes the light widget's position controls with the scene
// --------------------------------------------------------------------
void PointLightWidget::processInteractions()
{
    if (m_dirty)
    {
        m_dirty = false; // Reset tracking before read sequence //

        double latitude  = ui->doubleSpinBox_latitude->value() / 180.0 * M_PI;
        double longitude = ui->doubleSpinBox_longitude->value() / 180.0 * M_PI;
        double distance  = ui->doubleSpinBox_distance->value();

        float cl = cos(latitude);

        m_light->setPositionX(cl * cos(longitude) * distance);
        m_light->setPositionY(sin(latitude)       * distance);
        m_light->setPositionZ(cl * sin(longitude) * distance);

        // m_light->setColor(ui-> ... ); // :TODO: 
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

    m_dirty = true;
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

    m_dirty = true;
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
    
    m_dirty = true;
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
    
    m_dirty = true;
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
    
    m_dirty = true;
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
    
    m_dirty = true;
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
    
    m_dirty = true;
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
    
    m_dirty = true;
}