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
#include "animateview.h"
#include "Actions/AddRemKeyAct.h"
#include "VoxLib/Action/ActionManager.h"
#include "VoxLib/Video/VidStream.h"
#include "VoxScene/Camera.h"
#include "VoxScene/PrimGroup.h"
#include "VoxScene/Light.h"
#include "VoxScene/RenderParams.h"

using namespace vox;

// ----------------------------------------------------------------------------
//  Constuctor - Connect widget slots and signals
// ----------------------------------------------------------------------------
AnimateWidget::AnimateWidget(QWidget * parent) : 
	QWidget(parent), 
    ui(new Ui::AnimateWidget)
{
	ui->setupUi(this);

	m_animateView = new AnimateView(this);
	ui->frameLayout->addWidget(m_animateView, 0, 0, 1, 1 );

    connect(MainWindow::instance, SIGNAL(sceneChanged(vox::Scene &,void *)), 
            this, SLOT(sceneChanged(vox::Scene &,void *)), Qt::DirectConnection);
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
void AnimateWidget::sceneChanged(Scene & scene, void * userInfo)
{
    // Register the change event callbacks 
    if (userInfo == MainWindow::instance)
    {
        auto animator = MainWindow::instance->scene()->animator;
        if (!animator) return;

        ui->spinBox_framerate->setValue((int)animator->framerate());

        animator->onAdd(std::bind(&AnimateWidget::onAddKey, this, 
            std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
        animator->onRemove(std::bind(&AnimateWidget::onRemoveKey, this, 
            std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
        
        updateControls();
    }
}

// ----------------------------------------------------------------------------
//  Applies widget control changes to the scene 
// ----------------------------------------------------------------------------
void AnimateWidget::onAddKey(int index, std::shared_ptr<KeyFrame> key, bool suppress)
{
    if (!suppress) ActionManager::instance().push(AddRemKeyAct::create(index, key, true));
    m_animateView->update();
    updateControls();
}

// ----------------------------------------------------------------------------
//  Applies widget control changes to the scene 
// ----------------------------------------------------------------------------
void AnimateWidget::onRemoveKey(int index, std::shared_ptr<KeyFrame> key, bool suppress)
{
    if (!suppress) ActionManager::instance().push(AddRemKeyAct::create(index, key, false));
    m_animateView->update();
    updateControls();
}

// ----------------------------------------------------------------------------
//  Updates the enabled status of the animation control buttons as necessary
// ----------------------------------------------------------------------------
void AnimateWidget::updateControls()
{
    auto scene = MainWindow::instance->scene();
    if (!scene) return;
    auto animator = scene->animator;
    if (!animator) return;
    auto keys = animator->keyframes();

    if (!keys.size())
    { 
        ui->pushButton_load->setEnabled(false);
        ui->pushButton_prev->setEnabled(false);
        ui->pushButton_next->setEnabled(false);
        ui->pushButton_first->setEnabled(false);
        ui->pushButton_last->setEnabled(false);
    }
    else
    {
        ui->pushButton_load->setEnabled(true);
        ui->pushButton_prev->setEnabled(true);
        ui->pushButton_next->setEnabled(true);
        ui->pushButton_first->setEnabled(true);
        ui->pushButton_last->setEnabled(true);
    }

    if (keys.size() < 2)
    {
        ui->pushButton_preview->setEnabled(false);
        ui->pushButton_render->setEnabled(false);
    }
    else
    {
        ui->pushButton_preview->setEnabled(true);
        ui->pushButton_render->setEnabled(true);
    }

    auto iter = keys.begin();
    while (iter != keys.end() && iter->first < ui->spinBox_frame->value()) ++iter;
    ui->pushButton_delete->setEnabled(
        iter != keys.end() && iter->first == ui->spinBox_frame->value());
}

// ----------------------------------------------------------------------------
//  Sets the current frame number
// ----------------------------------------------------------------------------
void AnimateWidget::setFrame(int value)
{
    ui->spinBox_frame->setValue(value);
    updateControls();
}

// ----------------------------------------------------------------------------
//  Sets the frame number in the hover frame display
// ----------------------------------------------------------------------------
void AnimateWidget::setFrameHover(int value)
{
    if (value != -1) 
        ui->label_frame->setText(boost::lexical_cast<String>(value).c_str());
    else ui->label_frame->clear();

}

// ----------------------------------------------------------------------------
//  Updates the graphics display when the frame index is changed
// ----------------------------------------------------------------------------
void AnimateWidget::on_spinBox_frame_valueChanged(int value)
{
    m_animateView->setFrame(value);
    updateControls();
}

// ----------------------------------------------------------------------------
//  Updates the graphics display when the frame index is changed
// ----------------------------------------------------------------------------
void AnimateWidget::on_spinBox_framerate_valueChanged(int value)
{
    MainWindow::instance->scene()->animator->setFramerate((unsigned int)value);
}

// ----------------------------------------------------------------------------
//  Inserts a new keyframe into the animator
// ----------------------------------------------------------------------------
void AnimateWidget::on_pushButton_key_clicked()
{
    auto & scene = MainWindow::instance->scene();
    auto frame = ui->spinBox_frame->value();
    scene->animator->addKeyframe(scene->generateKeyFrame(), frame);
}

// ----------------------------------------------------------------------------
//  Loads the current keyframe (or interpolated frame) into the scene
// ----------------------------------------------------------------------------
void AnimateWidget::on_pushButton_load_clicked()
{
    auto & scene  = MainWindow::instance->scene();
    auto animator = scene->animator;

    auto index  = ui->spinBox_frame->value();
    auto frames = scene->animator->keyframes();
    if (frames.empty()) return;

    MainWindow::instance->stopRender();
    
    auto lock = scene->lock(MainWindow::instance); // Lock for edit

    // Load the endpoint if the selected frame is outside the existing range of frame values
    if      (frames.front().first > index) frames.front().second->clone(scene);
    else if (frames.back().first  < index) frames.back().second->clone(scene);
    else
    {
        auto iter = frames.begin();
        while (iter->first < index) iter++;
        
        // If on an actual frame then load it ...
        if (iter->first == index)
        {
            scene = iter->second->clone(scene);
        }
        else // ... interpolate to the selected frame
        {
            auto  fend = iter->second;
            float tend = iter->first;
            --iter;
            auto fbeg = iter->second;
            float tbeg = iter->first;
        
            float factor = (index - tbeg) / (tend - tbeg);

            scene->animator->interp(fbeg, fend, factor, scene);
        }
    }

    if (scene->parameters)   scene->parameters->setDirty();
    if (scene->lightSet)     scene->lightSet->setDirty();
    if (scene->camera)       scene->camera->setDirty();
    if (scene->clipGeometry) scene->clipGeometry->setDirty();
    if (scene->transfer)     scene->transfer->setDirty();
    if (scene->transferMap)  scene->transferMap->setDirty();

    lock.reset();

    MainWindow::instance->beginRender();
}

// ----------------------------------------------------------------------------
//  Keyframe seeking buttons
// ----------------------------------------------------------------------------
void AnimateWidget::on_pushButton_next_clicked() 
{ 
    auto currFrame = ui->spinBox_frame->value();

    auto animator = MainWindow::instance->scene()->animator;
    auto frames = animator->keyframes();
    auto iter = frames.begin();
    while (iter != frames.end() && iter->first <= currFrame) ++iter;
    if (iter != frames.end()) ui->spinBox_frame->setValue(iter->first);

    updateControls();
}
void AnimateWidget::on_pushButton_prev_clicked()
{ 
    auto currFrame = ui->spinBox_frame->value();
    
    auto animator = MainWindow::instance->scene()->animator;
    auto frames = animator->keyframes();
    if (frames.empty() || frames.front().first >= currFrame) return;
    auto iter = frames.begin();
    while (iter != frames.end() && iter->first < currFrame) ++iter;
    --iter;
    if (iter != frames.end()) ui->spinBox_frame->setValue(iter->first);
    
    updateControls();
}
void AnimateWidget::on_pushButton_first_clicked()
{ 
    auto animator = MainWindow::instance->scene()->animator;
    auto frames = animator->keyframes();
    if (frames.empty()) return;

    ui->spinBox_frame->setValue(frames.front().first);
    
    updateControls();
}
void AnimateWidget::on_pushButton_last_clicked()
{ 
    auto animator = MainWindow::instance->scene()->animator;
    auto frames = animator->keyframes();
    if (frames.empty()) return;

    ui->spinBox_frame->setValue(frames.back().first);
    
    updateControls();
}

// ----------------------------------------------------------------------------
//  Deletes a keyframe from the animator 
// ----------------------------------------------------------------------------
void AnimateWidget::on_pushButton_delete_clicked()
{
    auto & scene = MainWindow::instance->scene();
    auto frame = ui->spinBox_frame->value();
    scene->animator->removeKeyframe(frame);
}

// ----------------------------------------------------------------------------
//  Begins rendering an animation sequence given the animation info
// ----------------------------------------------------------------------------
void AnimateWidget::on_pushButton_preview_clicked()
{
    MainWindow::instance->scene()->animator->setOutputUri("");

    MainWindow::instance->stopRender();
    MainWindow::instance->beginRender(ui->spinBox_samples->value(), true);
}

// ----------------------------------------------------------------------------
//  Begins rendering an animation sequence given the animation info
// ----------------------------------------------------------------------------
void AnimateWidget::on_pushButton_render_clicked()
{
    // Detect available export types from AV exporters
    String fileTypes;
    auto encoders = VidOStream::encoders();
    BOOST_FOREACH (auto & encoder, encoders)
        fileTypes += format("(*%1%)\n", encoder);

    QString filename = QFileDialog::getSaveFileName( 
        this, tr("Choose an video destination"), QString(), 
        fileTypes.c_str());
    
    if (filename.isEmpty()) return;

    ResourceId uri(("file:///" + filename).toUtf8().data());
    MainWindow::instance->scene()->animator->setOutputUri(uri);

    MainWindow::instance->stopRender();
    MainWindow::instance->beginRender(ui->spinBox_samples->value(), true);
}