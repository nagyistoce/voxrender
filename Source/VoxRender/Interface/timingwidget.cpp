/* ===========================================================================

	Project: VoxRender

	Description: Implements a control interface for 4D volume time steps

    Copyright (C) 2014 Lucas Sherman

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
#include "ui_timingwidget.h"
#include "timingwidget.h"

// Include Dependencies
#include "mainwindow.h"
#include "utilities.h"

// VoxScene Bundles
#include "VoxScene/Volume.h"

using namespace vox;

// ----------------------------------------------------------------------------
//  Constuctor - Connect widget slots and signals
// ----------------------------------------------------------------------------
TimingWidget::TimingWidget(QWidget *parent) : 
	QWidget(parent), 
    ui(new Ui::TimingWidget),
    m_ignore(false)
{
	ui->setupUi(this);

    connect(MainWindow::instance, SIGNAL(sceneChanged(vox::Scene &,void *)), 
            this, SLOT(sceneChanged(vox::Scene &,void *)), Qt::DirectConnection);
}
    
// ----------------------------------------------------------------------------
//  Clear UI
// ----------------------------------------------------------------------------
TimingWidget::~TimingWidget()
{
    delete ui;
}

// ----------------------------------------------------------------------------
//  Synchronizes the widget controls with the current scene 
// ----------------------------------------------------------------------------
void TimingWidget::sceneChanged(Scene & scene, void * userInfo)
{
    if (userInfo == this || !scene.volume) return;
    auto volume = scene.volume;

    m_ignore = true;

    // Configure the time step slider
    auto timeSteps = volume->extent()[3]-1;
    if (!timeSteps)
    {
        ui->spinBox_t->setEnabled(false);
        ui->horizontalSlider_t->setEnabled(false);
    }
    else
    {
        ui->spinBox_t->setValue((int)volume->timeSlice());
        ui->spinBox_t->setEnabled(true);
        ui->horizontalSlider_t->setEnabled(true);
        ui->spinBox_t->setMaximum(timeSteps);
        ui->horizontalSlider_t->setMaximum(timeSteps);
    }

    // Update the volume spacing controls
    auto spacing = volume->spacing();
    ui->doubleSpinBox_x->setValue(spacing[0]);
    ui->doubleSpinBox_y->setValue(spacing[1]);
    ui->doubleSpinBox_z->setValue(spacing[2]);

    m_ignore = false;
}

// ----------------------------------------------------------------------------
//  Applies widget control changes to the scene 
// ----------------------------------------------------------------------------
void TimingWidget::update()
{
    if (m_ignore) return;
    auto scene = MainWindow::instance->scene();
    auto volume = scene->volume;
    if (!volume) return;

    auto lock = scene->lock(this);

    Vector4f spacing = volume->spacing();
    spacing[0] = ui->doubleSpinBox_x->value();
    spacing[1] = ui->doubleSpinBox_y->value();
    spacing[2] = ui->doubleSpinBox_z->value();
    volume->setSpacing(spacing);

    volume->setTimeSlice(ui->spinBox_t->value());

    volume->setDirty();
}

// ----------------------------------------------------------------------------
//                  Widget Value Change Detection
// ----------------------------------------------------------------------------
void TimingWidget::on_horizontalSlider_t_valueChanged(int value)
{
    update();
}
void TimingWidget::on_horizontalSlider_x_valueChanged(int value)
{
    Utilities::forceSbToSl(
        ui->doubleSpinBox_x,
        ui->horizontalSlider_x,
        value);
    
    update();
}
void TimingWidget::on_doubleSpinBox_x_valueChanged(double value)
{
    Utilities::forceSlToSb(
        ui->horizontalSlider_x,
        ui->doubleSpinBox_x,
        value);
    
    update();
}
void TimingWidget::on_horizontalSlider_y_valueChanged(int value)
{
    Utilities::forceSbToSl(
        ui->doubleSpinBox_y,
        ui->horizontalSlider_y,
        value);
    
    update();
}
void TimingWidget::on_doubleSpinBox_y_valueChanged(double value)
{
    Utilities::forceSlToSb(
        ui->horizontalSlider_y,
        ui->doubleSpinBox_y,
        value);
    
    update();
}
void TimingWidget::on_horizontalSlider_z_valueChanged(int value)
{
    Utilities::forceSbToSl(
        ui->doubleSpinBox_z,
        ui->horizontalSlider_z,
        value);
    
    update();
}
void TimingWidget::on_doubleSpinBox_z_valueChanged(double value)
{
    Utilities::forceSlToSb(
        ui->horizontalSlider_z,
        ui->doubleSpinBox_z,
        value);
    
    update();
}