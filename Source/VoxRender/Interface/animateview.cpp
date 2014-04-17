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

// Include Header
#include "animateview.h"

// Include Dependencies
#include "mainwindow.h"

using namespace vox;

// ----------------------------------------------------------------------------
//  Constructor - Initialize display scene and view parameters
// ----------------------------------------------------------------------------
AnimateView::AnimateView(AnimateWidget * parent) : 
    m_animateItem(new AnimateItem(parent))
{
	// Make scene backdrop flush with window color
 	setFrameShadow(Sunken); setFrameShape(NoFrame);
	setBackgroundBrush(QBrush(QColor(240, 240, 240)));

	// Set interaction policies
	setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	setViewportUpdateMode(QGraphicsView::FullViewportUpdate);
	setDragMode(QGraphicsView::ScrollHandDrag);

	// Setup histogram scene
	m_scene.setBackgroundBrush(QColor(255, 255, 255));
	m_scene.addItem(m_animateItem); 
	setScene(&m_scene);
    setMouseTracking(true);
}
    
// ----------------------------------------------------------------------------
// Destructor - Frees the image buffer
// ----------------------------------------------------------------------------
AnimateView::~AnimateView()
{
}

// ----------------------------------------------------------------------------
//  
// ----------------------------------------------------------------------------
void AnimateView::mouseMoveEvent(QMouseEvent* event)
{
    m_animateItem->onMouseMove(event);
}

// ----------------------------------------------------------------------------
//  Zoom in/out on mouse wheel event
// ----------------------------------------------------------------------------
void AnimateView::wheelEvent(QWheelEvent* event) 
{
}

// ----------------------------------------------------------------------------
//  Histogram view resize event
// ----------------------------------------------------------------------------
void AnimateView::resizeEvent(QResizeEvent *event) 
{	
    // Resize the canvas rectangle and compute margins
	m_canvasRectangle = rect();

	m_scene.setSceneRect(m_canvasRectangle);

	m_canvasRectangle.adjust(0, 0, -1, -20);

	m_animateItem->setRect(m_canvasRectangle);

    //m_histogramItem.setOffset(m_canvasRectangle.left(), m_canvasRectangle.top());

	QGraphicsView::resizeEvent(event);
}