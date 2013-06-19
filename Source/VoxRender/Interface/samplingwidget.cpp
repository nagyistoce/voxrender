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
#include "VoxLib/Scene/RenderParams.h"

// ----------------------------------------------------------------------------
//  Constuctor - Connect widget slots and signals
// ----------------------------------------------------------------------------
SamplingWidget::SamplingWidget(QWidget *parent) : 
	QWidget(parent), 
    ui(new Ui::SamplingWidget)
{
	ui->setupUi(this);
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
void SamplingWidget::synchronizeView()
{
    // Synchronize the camera object controls
    vox::RenderParams & settings = *MainWindow::instance->scene().parameters;
    
    ui->doubleSpinBox_primaryStep->setValue( (double)settings.primaryStepSize() );
    ui->doubleSpinBox_shadowStep->setValue ( (double)settings.shadowStepSize()  );
    ui->doubleSpinBox_occludeStep->setValue( (double)settings.occludeStepSize() );
    
    ui->doubleSpinBox_gradient->setValue( (double)settings.gradientCutoff() );

    ui->spinBox_occludeSamples->setValue( (int)settings.occludeSamples() );

    m_dirty = false;
}

// ----------------------------------------------------------------------------
//  Applies widget control changes to the scene 
// ----------------------------------------------------------------------------
void SamplingWidget::processInteractions()
{
    if (m_dirty)
    {
        m_dirty = false;

        vox::RenderParams & settings = *MainWindow::instance->scene().parameters;
    
        settings.setPrimaryStepSize( (float)ui->doubleSpinBox_primaryStep->value() );
        settings.setShadowStepSize ( (float)ui->doubleSpinBox_shadowStep->value()  );
        settings.setOccludeStepSize( (float)ui->doubleSpinBox_occludeStep->value() );
        settings.setOccludeSamples( (unsigned int)ui->spinBox_occludeSamples->value() );

        settings.setGradientCutoff( (float)ui->doubleSpinBox_gradient->value() );
    }
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
    
    m_dirty = true;
}
void SamplingWidget::on_doubleSpinBox_primaryStep_valueChanged(double value)
{
    Utilities::forceSlToSb(
        ui->horizontalSlider_primaryStep,
        ui->doubleSpinBox_primaryStep,
        value);
    
    m_dirty = true;
}
void SamplingWidget::on_horizontalSlider_shadowStep_valueChanged(int value)
{
    Utilities::forceSbToSl(
        ui->doubleSpinBox_shadowStep,
        ui->horizontalSlider_shadowStep,
        value);
    
    m_dirty = true;
}
void SamplingWidget::on_doubleSpinBox_shadowStep_valueChanged(double value)
{
    Utilities::forceSlToSb(
        ui->horizontalSlider_shadowStep,
        ui->doubleSpinBox_shadowStep,
        value);
    
    m_dirty = true;
}
void SamplingWidget::on_horizontalSlider_occludeStep_valueChanged(int value)
{
    Utilities::forceSbToSl(
        ui->doubleSpinBox_occludeStep,
        ui->horizontalSlider_occludeStep,
        value);
    
    m_dirty = true;
}
void SamplingWidget::on_doubleSpinBox_occludeStep_valueChanged(double value)
{
    Utilities::forceSlToSb(
        ui->horizontalSlider_occludeStep,
        ui->doubleSpinBox_occludeStep,
        value);
    
    m_dirty = true;
}
void SamplingWidget::on_horizontalSlider_occludeSamples_valueChanged(int value) 
{ 
    m_dirty = true; 
}
void SamplingWidget::on_horizontalSlider_gradient_valueChanged(int value) 
{ 
    Utilities::forceSbToSl(
        ui->doubleSpinBox_gradient,
        ui->horizontalSlider_gradient,
        value);

    m_dirty = true; 
}
void SamplingWidget::on_doubleSpinBox_gradient_valueChanged(double value)
{
    Utilities::forceSlToSb(
        ui->horizontalSlider_gradient,
        ui->doubleSpinBox_gradient,
        value);
    
    m_dirty = true;
}