/* ===========================================================================

	Project: VoxRender

	Description: Volume rendering application

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

// Begin definition
#ifndef ANIMATE_VIEW_H
#define ANIMATE_VIEW_H

// Include Dependencies
#include "griditem.h"

// QT Dependencies
#include <QtCore/QEvent>
#include <QtCore/QMargins>
#include <QtCore/QPoint>
#include <QtGui/QWheelEvent>
#include <QtGui/QMouseEvent>
#include <QtGui/QMatrix>
#include <QtGui/QClipboard>
#include <QtWidgets/QApplication>
#include <QtWidgets/QGraphicsPixmapItem>
#include <QtWidgets/QGraphicsScene>
#include <QtWidgets/QGraphicsView>

/** Implements a QT graphics view for displaying animation sequence data */
class AnimateView : public QGraphicsView
{
	Q_OBJECT

public:
    ~AnimateView();

	AnimateView(QWidget * parent = 0);

private slots:
    void onSceneChanged();

private:
	QGraphicsScene m_scene; ///< Scene handle
};

// End definition
#endif // HISTOGRAM_VIEW_H