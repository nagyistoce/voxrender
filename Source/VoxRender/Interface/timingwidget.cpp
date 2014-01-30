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

    connect(MainWindow::instance, SIGNAL(sceneChanged()), this, SLOT(sceneChanged()));
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
void TimingWidget::sceneChanged()
{
    // Synchronize the camera object controls
    auto volume = MainWindow::instance->scene().volume;
    if (!volume) return;

    m_ignore = true;

    m_ignore = false;
}

// ----------------------------------------------------------------------------
//  Applies widget control changes to the scene 
// ----------------------------------------------------------------------------
void TimingWidget::update()
{
    if (m_ignore) return;
}