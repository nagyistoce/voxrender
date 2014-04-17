/* ===========================================================================

	Project: VoxRender
    
	Description: Performs interactive rendering of volume data using 
		photon mapping and volume ray casting techniques.

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
#include "animateitem.h"

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

class AnimateWidget;

/** Implements a QT graphics view for volume histogram data and transfer functions */
class AnimateView : public QGraphicsView
{
	Q_OBJECT

public:
    ~AnimateView();

    /** Constructs a new HistogramView with an option transfer function editor */
	AnimateView(AnimateWidget * parent);

    void setFrame(int value) { m_animateItem->setFrame(value); }
    void update() { m_animateItem->update(); }

private:
    void mouseMoveEvent(QMouseEvent* event);
	void wheelEvent(QWheelEvent *event);
	void resizeEvent(QResizeEvent *event);
    
	AnimateItem *  m_animateItem;  ///< Animate image
	QGraphicsScene m_scene;        ///< Scene object

    QRectF   m_canvasRectangle; ///< Full canvas for drawing operations
	QMargins m_margins;         ///< Margins for actual histogram display
};

// End definition
#endif // ANIMATE_VIEW_H