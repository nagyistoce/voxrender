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

// Include Dependencies
#include "animateview.h"

// QT Includes
#include <QtWidgets/QWidget>

namespace Ui { class AnimateWidget; }

// Volume data histogram widget
class AnimateWidget : public QWidget
{
	Q_OBJECT

public:
	explicit AnimateWidget(QWidget *parent = 0);

	~AnimateWidget();

private:
	Ui::AnimateWidget *ui;

    void update();

    bool m_ignore;

    float m_timeOffset; ///< Time offset to start frame in window

    AnimateView * m_animateView;

signals:

private slots:
    void sceneChanged();

    void on_pushButton_addKey_clicked();
    void on_pushButton_deleteKey_clicked();
};

// End definition
#endif // ANIMATE_WIDGET_H

