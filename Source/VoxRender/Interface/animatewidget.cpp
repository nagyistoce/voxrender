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
#include "animateitem.h"

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

    ui->view->setScene(&m_scene);
    
    ui->view->setFrameShape(QGraphicsView::NoFrame);
    ui->view->setFrameShadow(QGraphicsView::Sunken);
	ui->view->setBackgroundBrush(QBrush(QColor(240, 240, 240)));
	ui->view->setVerticalScrollBarPolicy( Qt::ScrollBarAlwaysOff );
	ui->view->setHorizontalScrollBarPolicy( Qt::ScrollBarAlwaysOff );
	ui->view->setViewportUpdateMode( QGraphicsView::FullViewportUpdate );
    ui->view->setDragMode( QGraphicsView::ScrollHandDrag );
    
    m_animateItem = new AnimateItem(nullptr);
    m_scene.addItem(m_animateItem);
    resizeEvent(&QResizeEvent(size(), size()));

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
//  Resizes the animation view when the widget is resized
// ----------------------------------------------------------------------------
void AnimateWidget::resizeEvent(QResizeEvent *event) 
{	
    // Resize the canvas rectangle and compute margins
	auto canvasRectangle = ui->view->viewport()->rect();
    
	m_scene.setSceneRect(canvasRectangle);

    canvasRectangle.adjust(0, 0, -1, -20);

    m_animateItem->setRect(canvasRectangle);
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
void AnimateWidget::on_pushButton_key_clicked()
{
    auto & scene = MainWindow::instance->scene();

    auto frame = ui->spinBox_frame->value();
    scene.animator->addKeyframe(scene.generateKeyFrame(), frame);
}

// ----------------------------------------------------------------------------
//  Synchronizes the widget controls with the current scene 
// ----------------------------------------------------------------------------
void AnimateWidget::on_pushButton_delete_clicked()
{
    auto & scene = MainWindow::instance->scene();

    auto frame = ui->spinBox_frame->value();
    scene.animator->removeKeyframe(frame);
}

// ----------------------------------------------------------------------------
//  Begins rendering an animation sequence given the animation info
// ----------------------------------------------------------------------------
void AnimateWidget::on_pushButton_render_clicked()
{
    MainWindow::instance->stopRender();
    MainWindow::instance->beginRender(ui->spinBox_samples->value(), true);
}