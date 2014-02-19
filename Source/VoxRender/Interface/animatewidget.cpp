/* ===========================================================================

	Project: VoxRender

	Description: Implements an interface for for animation management

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
#include "ui_animatewidget.h"
#include "animatewidget.h"

// Include Dependencies
#include "mainwindow.h"
#include "utilities.h"

using namespace vox;

// ----------------------------------------------------------------------------
//  Constuctor - Connect widget slots and signals
// ----------------------------------------------------------------------------
AnimateWidget::AnimateWidget(QWidget * parent) : 
	QWidget(parent), 
    ui(new Ui::AnimateWidget),
    m_ignore(false)
{
	ui->setupUi(this);
    
    // Create the histogram view 
	m_animateView = new AnimateView(ui->frame_display);
	ui->animateLayout->addWidget(m_animateView, 0, 0, 1, 1);

    connect(MainWindow::instance, SIGNAL(sceneChanged()), this, SLOT(sceneChanged()));
}
    
// ----------------------------------------------------------------------------
//  Clear UI
// ----------------------------------------------------------------------------
AnimateWidget::~AnimateWidget()
{
    delete ui;
}

// ----------------------------------------------------------------------------
//  Synchronizes the widget controls with the current scene 
// ----------------------------------------------------------------------------
void AnimateWidget::sceneChanged()
{
    m_ignore = true;

    m_ignore = false;
}

// ----------------------------------------------------------------------------
//  Applies widget control changes to the scene 
// ----------------------------------------------------------------------------
void AnimateWidget::update()
{
}

// ----------------------------------------------------------------------------
//  Synchronizes the widget controls with the current scene 
// ----------------------------------------------------------------------------
void AnimateWidget::on_pushButton_addKey_clicked()
{
}

// ----------------------------------------------------------------------------
//  Synchronizes the widget controls with the current scene 
// ----------------------------------------------------------------------------
void AnimateWidget::on_pushButton_deleteKey_clicked()
{
}