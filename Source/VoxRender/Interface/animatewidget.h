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

// Begin Definition
#ifndef ANIMATE_WIDGET_H
#define ANIMATE_WIDGET_H

// QT Includes
#include <QtWidgets/QWidget>
#include <QtWidgets/QGraphicsScene>

#include "VoxScene/Animator.h"
#include "VoxScene/Scene.h"

namespace Ui { class AnimateWidget; }

class AnimateView;

// Volume data histogram widget
class AnimateWidget : public QWidget
{
	Q_OBJECT

public:
	explicit AnimateWidget(QWidget *parent = 0);

	~AnimateWidget();

    void setFrame(int value);

    void setFrameHover(int value);

private:
	Ui::AnimateWidget * ui;

    void update();

    void onAddKey(int index, vox::KeyFrame & key, bool suppress);
    void onRemoveKey(int index, vox::KeyFrame & key, bool suppress);

    bool m_ignore;

    float m_frameOffset; ///< Frame at left edge of window

    QGraphicsScene m_scene;

    AnimateView * m_animateView;

private slots:
    void sceneChanged();

    void on_spinBox_frame_valueChanged(int value);
    void on_spinBox_framerate_valueChanged(int value);

    void on_pushButton_render_clicked();
    void on_pushButton_preview_clicked();
    void on_pushButton_key_clicked();
    void on_pushButton_delete_clicked();
    void on_pushButton_load_clicked();

    void on_pushButton_next_clicked();
    void on_pushButton_prev_clicked();
    void on_pushButton_first_clicked();
    void on_pushButton_last_clicked();
};

// End definition
#endif // ANIMATE_WIDGET_H

