/* ===========================================================================

	Project: SamplingWidget - Sampling widget

	Description:
	 Implements an interface for modifying the sampling parameters of the
	 volume ray tracer.

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
#include "ui_samplingwidget.h"
#include "samplingwidget.h"

// Include Dependencies
#include "mainwindow.h"
#include "utilities.h"

// VoxRender Dependencies
#include "VoxScene/RenderParams.h"

using namespace vox;

// ----------------------------------------------------------------------------
//  Constuctor - Connect widget slots and signals
// ----------------------------------------------------------------------------
SamplingWidget::SamplingWidget(QWidget *parent) : 
	QWidget(parent), 
    ui(new Ui::SamplingWidget),
    m_ignore(false)
{
	ui->setupUi(this);

    connect(MainWindow::instance, SIGNAL(sceneChanged(vox::Scene &,void *)), 
            this, SLOT(sceneChanged(vox::Scene &,void *)), Qt::DirectConnection);
}
    
// ----------------------------------------------------------------------------
//  Clear UI
// ----------------------------------------------------------------------------
SamplingWidget::~SamplingWidget()
{
    delete ui;
}

// ----------------------------------------------------------------------------
//  Synchronizes the widget controls with the current scene 
// ----------------------------------------------------------------------------
void SamplingWidget::sceneChanged(Scene & scene, void * userInfo)
{
    if (userInfo == this || !scene.parameters) return;
    auto settings = scene.parameters;

    m_ignore = true;

    ui->doubleSpinBox_primaryStep->setValue((double)settings->primaryStepSize());
    ui->doubleSpinBox_shadowStep->setValue ((double)settings->shadowStepSize());
    ui->doubleSpinBox_coefficient->setValue((double)settings->scatterCoefficient());
    
    ui->doubleSpinBox_gradient->setValue((double)settings->gradientCutoff());

    m_ignore = false;
}

// ----------------------------------------------------------------------------
//  Applies widget control changes to the scene 
// ----------------------------------------------------------------------------
void SamplingWidget::update()
{
    if (m_ignore) return;

    auto scene = MainWindow::instance->scene();
    auto settings = scene->parameters;
    if (!settings) return;

    auto lock = scene->lock(this);

    settings->setPrimaryStepSize( (float)ui->doubleSpinBox_primaryStep->value() );
    settings->setShadowStepSize ( (float)ui->doubleSpinBox_shadowStep->value()  );
    settings->setScatterCoefficient( (float)ui->doubleSpinBox_coefficient->value() );
    settings->setEdgeEnhancement( (float)ui->doubleSpinBox_edge->value() );
    settings->setGradientCutoff( (float)ui->doubleSpinBox_gradient->value() );
    settings->setDirty();
}

// ----------------------------------------------------------------------------
//                  Widget Value Change Detection
// ----------------------------------------------------------------------------
void SamplingWidget::on_horizontalSlider_primaryStep_valueChanged(int value)
{
    Utilities::forceSbToSl(
        ui->doubleSpinBox_primaryStep,
        ui->horizontalSlider_primaryStep,
        value);
    
    update();
}
void SamplingWidget::on_doubleSpinBox_primaryStep_valueChanged(double value)
{
    Utilities::forceSlToSb(
        ui->horizontalSlider_primaryStep,
        ui->doubleSpinBox_primaryStep,
        value);
    
    update();
}
void SamplingWidget::on_horizontalSlider_shadowStep_valueChanged(int value)
{
    Utilities::forceSbToSl(
        ui->doubleSpinBox_shadowStep,
        ui->horizontalSlider_shadowStep,
        value);
    
    update();
}
void SamplingWidget::on_doubleSpinBox_shadowStep_valueChanged(double value)
{
    Utilities::forceSlToSb(
        ui->horizontalSlider_shadowStep,
        ui->doubleSpinBox_shadowStep,
        value);
    
    update();
}
void SamplingWidget::on_horizontalSlider_gradient_valueChanged(int value) 
{ 
    Utilities::forceSbToSl(
        ui->doubleSpinBox_gradient,
        ui->horizontalSlider_gradient,
        value);
    
    update();
}
void SamplingWidget::on_doubleSpinBox_gradient_valueChanged(double value)
{
    Utilities::forceSlToSb(
        ui->horizontalSlider_gradient,
        ui->doubleSpinBox_gradient,
        value);
    
    update();
}
void SamplingWidget::on_horizontalSlider_coefficient_valueChanged(int value) 
{ 
    Utilities::forceSbToSl(
        ui->doubleSpinBox_coefficient,
        ui->horizontalSlider_coefficient,
        value);
    
    update();
}
void SamplingWidget::on_doubleSpinBox_coefficient_valueChanged(double value)
{
    Utilities::forceSlToSb(
        ui->horizontalSlider_coefficient,
        ui->doubleSpinBox_coefficient,
        value);
    
    update();
}
void SamplingWidget::on_horizontalSlider_edge_valueChanged(int value) 
{ 
    Utilities::forceSbToSl(
        ui->doubleSpinBox_edge,
        ui->horizontalSlider_edge,
        value);
    
    update();
}
void SamplingWidget::on_doubleSpinBox_edge_valueChanged(double value)
{
    Utilities::forceSlToSb(
        ui->horizontalSlider_edge,
        ui->doubleSpinBox_edge,
        value);
    
    update();
}