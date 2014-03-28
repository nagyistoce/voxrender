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
#include "Actions/AddRemKeyAct.h"
#include "VoxLib/Action/ActionManager.h"

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

    ui->view->setMouseTracking(true);
    ui->view->setFrameShape(QGraphicsView::NoFrame);
    ui->view->setFrameShadow(QGraphicsView::Sunken);
	ui->view->setBackgroundBrush(QBrush(QColor(240, 240, 240)));
	ui->view->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	ui->view->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	ui->view->setViewportUpdateMode(QGraphicsView::FullViewportUpdate);
    
    m_animateItem = new AnimateItem(this);
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

    QWidget::resizeEvent(event);
}

// ----------------------------------------------------------------------------
//  Synchronizes the widget controls with the current scene 
// ----------------------------------------------------------------------------
void AnimateWidget::sceneChanged()
{
    m_ignore = true;

    auto animator = MainWindow::instance->scene().animator;
    if (!animator) { m_ignore = false; return; }

    animator->onAdd(std::bind(&AnimateWidget::onAddKey, this, 
        std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
    animator->onRemove(std::bind(&AnimateWidget::onRemoveKey, this, 
        std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));

    m_ignore = false;
}

// ----------------------------------------------------------------------------
//  Applies widget control changes to the scene 
// ----------------------------------------------------------------------------
void AnimateWidget::update()
{
}

// ----------------------------------------------------------------------------
//  Applies widget control changes to the scene 
// ----------------------------------------------------------------------------
void AnimateWidget::onAddKey(int index, KeyFrame & key, bool suppress)
{
    if (!suppress) ActionManager::instance().push(AddRemKeyAct::create(index, key, true));
    m_animateItem->update();
}

// ----------------------------------------------------------------------------
//  Applies widget control changes to the scene 
// ----------------------------------------------------------------------------
void AnimateWidget::onRemoveKey(int index, KeyFrame & key, bool suppress)
{
    if (!suppress) ActionManager::instance().push(AddRemKeyAct::create(index, key, false));
    m_animateItem->update();
}

// ----------------------------------------------------------------------------
//  Sets the current frame number
// ----------------------------------------------------------------------------
void AnimateWidget::setFrame(int value)
{
    ui->spinBox_frame->setValue(value);
}

// ----------------------------------------------------------------------------
//  Updates the graphics display when the frame index is changed
// ----------------------------------------------------------------------------
void AnimateWidget::on_spinBox_frame_valueChanged(int value)
{
    m_animateItem->setFrame(value);
}

// ----------------------------------------------------------------------------
//  Inserts a new keyframe into the animator
// ----------------------------------------------------------------------------
void AnimateWidget::on_pushButton_key_clicked()
{
    auto & scene = MainWindow::instance->scene();
    auto frame = ui->spinBox_frame->value();
    scene.animator->addKeyframe(scene.generateKeyFrame(), frame);
}

// ----------------------------------------------------------------------------
//  Loads the current keyframe (or interpolated frame) into the scene
// ----------------------------------------------------------------------------
void AnimateWidget::on_pushButton_load_clicked()
{
    auto & scene  = MainWindow::instance->scene();
    auto animator = scene.animator;

    auto index  = ui->spinBox_frame->value();
    auto frames = scene.animator->keyframes();
    if (frames.empty()) return;

    MainWindow::instance->stopRender();

    if      (frames.front().first > index) frames.front().second.clone(scene);
    else if (frames.back().first  < index) frames.back().second.clone(scene);
    else
    {
        auto iter = frames.begin();
        while (iter->first < index) iter++;

        if (iter->first == index)
        {
            scene = iter->second;
        }
        else
        {
            auto  fend = iter->second;
            float tend = iter->first;
            --iter;
            auto fbeg = iter->second;
            float tbeg = iter->first;
        
            float factor = (index - tbeg) / (tend - tbeg);

            scene.reset();
            scene.animator->interp(fbeg, fend, scene, factor);

        }
    }

    scene.animator = animator;

    MainWindow::instance->beginRender();
}

// ----------------------------------------------------------------------------
//  Keyframe seeking buttons
// ----------------------------------------------------------------------------
void AnimateWidget::on_pushButton_next_clicked() 
{ 
    auto currFrame = ui->spinBox_frame->value();

    auto animator = MainWindow::instance->scene().animator;
    auto frames = animator->keyframes();
    auto iter = frames.begin();
    while (iter != frames.end() && iter->first <= currFrame) ++iter;
    if (iter != frames.end()) ui->spinBox_frame->setValue(iter->first);
}
void AnimateWidget::on_pushButton_prev_clicked()
{ 
    auto currFrame = ui->spinBox_frame->value();
    
    auto animator = MainWindow::instance->scene().animator;
    auto frames = animator->keyframes();
    if (frames.empty() || frames.front().first >= currFrame) return;
    auto iter = frames.begin();
    while (iter != frames.end() && iter->first < currFrame) ++iter;
    --iter;
    if (iter != frames.end()) ui->spinBox_frame->setValue(iter->first);
}
void AnimateWidget::on_pushButton_first_clicked()
{ 
    auto animator = MainWindow::instance->scene().animator;
    auto frames = animator->keyframes();
    if (frames.empty()) return;

    ui->spinBox_frame->setValue(frames.front().first);
}
void AnimateWidget::on_pushButton_last_clicked()
{ 
    auto animator = MainWindow::instance->scene().animator;
    auto frames = animator->keyframes();
    if (frames.empty()) return;

    ui->spinBox_frame->setValue(frames.back().first);
}

// ----------------------------------------------------------------------------
//  Deletes a keyframe from the animator 
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