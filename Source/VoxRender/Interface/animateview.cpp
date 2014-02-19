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

// Include Header
#include "animateview.h"

// Include Dependencies
#include "mainwindow.h"

using namespace vox;

// ----------------------------------------------------------------------------
//  Constructor - Initialize display scene and view parameters
// ----------------------------------------------------------------------------
AnimateView::AnimateView(QWidget * parent) : 
    QGraphicsView(parent)
{
	// Make scene backdrop flush with window color
 	setFrameShadow(Sunken); setFrameShape(NoFrame);
	setBackgroundBrush(QBrush(QColor(240, 240, 240)));

	// Set interaction policies
	setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
	setViewportUpdateMode(QGraphicsView::FullViewportUpdate);

    // Configure the slots'n'signals for histogram image generation and update
    connect(MainWindow::instance, SIGNAL(sceneChanged()), this, SLOT(onSceneChanged()));

	// Setup histogram scene
	m_scene.setBackgroundBrush(QColor(255, 255, 255));
	setScene(&m_scene);
}
    
// ---------------------------------------------------------
//  Destructor 
// ---------------------------------------------------------
AnimateView::~AnimateView()
{
}

// ---------------------------------------------------------
//  
// ---------------------------------------------------------
void AnimateView::onSceneChanged()
{
}