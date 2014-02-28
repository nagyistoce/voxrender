/* ===========================================================================

	Project: VoxRender 

	Description: Implements a control interface for ambient lighting

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
#include "ui_ambientlightwidget.h"
#include "ambientlightwidget.h"

// Include Dependencies
#include "utilities.h"
#include "mainwindow.h"

// VoxLib Dependencies
#include "VoxScene/Light.h"

using namespace vox;

// --------------------------------------------------------------------
//  Constructor - Initialize the widget ui
// --------------------------------------------------------------------
AmbientLightWidget::AmbientLightWidget(QWidget * parent) : 
    QWidget(parent), 
    ui(new Ui::AmbientLightWidget)
{
	ui->setupUi(this);

    m_colorButton = new QColorPushButton();
    ui->layout_colorButton->addWidget(m_colorButton);

    connect(m_colorButton, SIGNAL(currentColorChanged(const QColor&)), this, SLOT(colorChanged(const QColor&)));
    connect(MainWindow::instance, SIGNAL(sceneChanged()), this, SLOT(sceneChanged()));
}

// --------------------------------------------------------------------
//  Destructor
// --------------------------------------------------------------------
AmbientLightWidget::~AmbientLightWidget()
{
    delete ui;
}

// --------------------------------------------------------------------
//  Synchronizes the scene with the light widget's controls
// --------------------------------------------------------------------
void AmbientLightWidget::update()
{
    auto lightSet = MainWindow::instance->scene().lightSet;

    lightSet->lock();

        QColor   color = m_colorButton->getColor();
        Vector3f light = Vector3f(color.red()/255.0f, color.green()/255.0f, color.blue()/255.0f) * ui->doubleSpinBox_intensity->value();
        lightSet->setAmbientLight(light);
        lightSet->setDirty();

    lightSet->unlock();
}

// --------------------------------------------------------------------
//  Synchronizes the light widget's controls with the scene
// --------------------------------------------------------------------
void AmbientLightWidget::sceneChanged()
{
    auto & scene = MainWindow::instance->scene();
    if (!scene.lightSet) return;

    Vector3f ambient = scene.lightSet->ambientLight();
    float magnitude = ambient.fold(high);
    ambient = magnitude ? ambient * 255.0f / magnitude : Vector3f(255.0f, 255.0f, 255.0f);
    ui->doubleSpinBox_intensity->setValue(magnitude);
    m_colorButton->setColor( QColor((int)ambient[0], (int)ambient[1], (int)ambient[2]), true);
}

// --------------------------------------------------------------------
//  Modify the associated double spinbox to reflect slide value change
// --------------------------------------------------------------------
void AmbientLightWidget::on_horizontalSlider_intensity_valueChanged(int value)
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
void AmbientLightWidget::on_doubleSpinBox_intensity_valueChanged(double value)
{
    Utilities::forceSlToSb(
        ui->horizontalSlider_intensity,
        ui->doubleSpinBox_intensity,
        value);
    
    update();
}

// --------------------------------------------------------------------
//  Signals a color change in the color selection widget
// --------------------------------------------------------------------
void AmbientLightWidget::colorChanged(QColor const& color)
{
    update();
}