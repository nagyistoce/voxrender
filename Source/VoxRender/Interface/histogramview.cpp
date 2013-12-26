/* ===========================================================================

	Project: HistogramView - Graphics view of histogram data

	Description: Implements a display interface for volume histograms

    Copyright (C) 2012 Lucas Sherman

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
#include "histogramview.h"

// Include Dependencies
#include "mainwindow.h"
#include "histogramgenerator.h"
#include "transferitem.h"

using namespace vox;

// ----------------------------------------------------------------------------
//  Constructor - Initialize display scene and view parameters
// ----------------------------------------------------------------------------
HistogramView::HistogramView(QWidget *parent, bool createTransferView) : 
    QGraphicsView(parent),
    m_type(DataType_Density),
	m_margins(30, 20, -20, -30),
    m_transferItem(nullptr),
    m_histogramItem(nullptr),
    m_gridItem(nullptr),
    m_options( 0 )
{
	// Make scene backdrop flush with window color
 	setFrameShadow(Sunken); setFrameShape(NoFrame);
	setBackgroundBrush(QBrush(QColor(240, 240, 240)));

	// Set interaction policies
	setVerticalScrollBarPolicy( Qt::ScrollBarAlwaysOff );
	setHorizontalScrollBarPolicy( Qt::ScrollBarAlwaysOff );
	setViewportUpdateMode( QGraphicsView::FullViewportUpdate );
	setDragMode( QGraphicsView::ScrollHandDrag );

    // Configure the slots'n'signals for histogram image generation and update
    connect(MainWindow::instance, SIGNAL(sceneChanged()), this, SLOT(updateHistogramData()));
    auto generator = HistogramGenerator::instance();
    connect(generator, SIGNAL(histogramComplete(int)), this, SLOT(onHistogramReady(int)));

	// Setup histogram scene
	m_scene.setBackgroundBrush( QColor(255, 255, 255) );
	m_scene.addItem( &m_histogramItem ); 
	m_scene.addItem( &m_gridItem );
	setScene( &m_scene );

    // Ensure correct ordering
	m_gridItem.setZValue( 0 );
	m_histogramItem.setZValue( 1 );
    
    // Transfer function item is optional...
    if (createTransferView)
    {
        m_transferItem = new TransferItem();
        m_scene.addItem( m_transferItem );
        m_transferItem->setZValue( 2 );
    }
}
    
// ---------------------------------------------------------
// Destructor - Frees the image buffer
// ---------------------------------------------------------
HistogramView::~HistogramView()
{
}

// ---------------------------------------------------------
// Zoom in/out on mouse wheel event
// ---------------------------------------------------------
void HistogramView::wheelEvent(QWheelEvent* event) 
{
	const float zoomsteps[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
	size_t numsteps = sizeof(zoomsteps) / sizeof(*zoomsteps);

	size_t index = std::min<size_t>(std::upper_bound(zoomsteps, zoomsteps + numsteps, zoomfactor) - zoomsteps, numsteps-1);
	if (event->delta( ) < 0) 
	{
		// if zoomfactor is equal to zoomsteps[index-1] we need index-2
		while (index > 0 && zoomsteps[--index] == zoomfactor);		
	}
	zoomfactor = zoomsteps[index];

	resetTransform( );
	scale( zoomfactor, zoomfactor );
}

// ----------------------------------------------------------------------------
//  Indicates a change in the displayed histogram type
// ----------------------------------------------------------------------------
void HistogramView::setType(int dataType)
{
    if (m_type == dataType) return;

    m_type = dataType;

    updateImage();
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
void HistogramView::onHistogramReady(int dataType)
{
    if (dataType == m_type) updateImage();
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
void HistogramView::updateImage()
{
    auto  generator = HistogramGenerator::instance();
    QSize size      = QSize(m_canvasRectangle.width(), m_canvasRectangle.height());

    switch (m_type)
    {
    case DataType_Density:
        m_histogramItem.setPixmap(generator->densityPixmap().scaled(size));
        break;

    case DataType_DensityGrad:
        m_histogramItem.setPixmap(generator->gradientPixmap().scaled(size));
        break;

    default:
        VOX_LOG_ERROR(Error_Bug, "GUI", format("Unrecognized histogram type: %1%", m_type));
    }
}

// ----------------------------------------------------------------------------
//  Updates the transfer function item for this view (if enabled)
// ----------------------------------------------------------------------------
void HistogramView::updateTransfer() 
{ 
    if (m_transferItem) 
    { 
        m_transferItem->synchronizeView(); 
    }
}

// ---------------------------------------------------------
// Updates the histogram data buffer
// ---------------------------------------------------------
void HistogramView::updateHistogramData()
{
    // :TODO: Enable loading animation in place of histogram
}

// ----------------------------------------------------------------------------
//  Histogram view resize event
// ----------------------------------------------------------------------------
void HistogramView::resizeEvent(QResizeEvent *event) 
{	
    // Resize the canvas rectangle and compute margins
	m_canvasRectangle = rect();

	m_scene.setSceneRect(m_canvasRectangle);

	m_canvasRectangle.adjust( 
        m_margins.left(), m_margins.top(), 
		m_margins.right(), m_margins.bottom() 
        );

    if (m_transferItem) 
    {
        m_transferItem->setRect(m_canvasRectangle);

        m_transferItem->onResizeEvent();
    }

	m_gridItem.setRect(m_canvasRectangle);

    m_histogramItem.setOffset(m_canvasRectangle.left(), m_canvasRectangle.top());

    updateImage();

	QGraphicsView::resizeEvent(event);
}