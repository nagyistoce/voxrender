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

using namespace vox;

// ---------------------------------------------------------
// Constructor - Initialize display scene and view parameters
// ---------------------------------------------------------
HistogramView::HistogramView( QWidget *parent ) : 
    QGraphicsView(parent),
    //m_type( RenderController::VolumeHistogramType_Density ),
	m_margins( 30, 20, -20, -30 ),
    m_imagebuffer( nullptr), 
    m_transferItem( nullptr ),
    m_histogramItem( nullptr ),
    m_gridItem( nullptr ),
    m_options( 0 ),
    m_binMax( 0 )
{
	// Make scene backdrop flush with window color
 	setFrameShadow(Sunken); setFrameShape(NoFrame);
	setBackgroundBrush(QBrush(QColor(240, 240, 240)));

	// Set interaction policies
	setVerticalScrollBarPolicy( Qt::ScrollBarAlwaysOff );
	setHorizontalScrollBarPolicy( Qt::ScrollBarAlwaysOff );
	setViewportUpdateMode( QGraphicsView::FullViewportUpdate );
	setDragMode( QGraphicsView::ScrollHandDrag );
    
    // Connect scene change signal to the image update slot
    connect( MainWindow::instance, SIGNAL(sceneChanged()), 
		this, SLOT(updateHistogramData()) );

	// Setup histogram scene
	m_scene.setBackgroundBrush( QColor(255, 255, 255) );
    m_scene.addItem( &m_transferItem );
	m_scene.addItem( &m_histogramItem ); 
	m_scene.addItem( &m_gridItem );
	setScene( &m_scene );

    // Ensure correct ordering
	m_gridItem.setZValue( 0 );
	m_histogramItem.setZValue( 1 );
    m_transferItem.setZValue( 2 );

	// Initialize image buffer
	updateHistogramImage( );
}
    
// ---------------------------------------------------------
// Destructor - Frees the image buffer
// ---------------------------------------------------------
HistogramView::~HistogramView( )
{
    delete m_imagebuffer;
}

// ---------------------------------------------------------
// Zoom in/out on mouse wheel event
// ---------------------------------------------------------
void HistogramView::wheelEvent( QWheelEvent* event ) 
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

// ---------------------------------------------------------
// Histogram view resize event
// ---------------------------------------------------------
void HistogramView::resizeEvent( QResizeEvent *event ) 
{	
	QGraphicsView::resizeEvent( event );

    // Update the histogram rectangle
	m_canvasRectangle = rect( );
	QGraphicsView::setSceneRect( m_canvasRectangle );
	m_canvasRectangle.adjust( m_margins.left( ), m_margins.top( ), 
		m_margins.right( ), m_margins.bottom( ) );

    // Resize the histogram grid item
    m_transferItem.setRect( m_canvasRectangle );
	m_gridItem.setRect( m_canvasRectangle );

    // Redraw the histogram
    updateHistogramImage( );
}

// ---------------------------------------------------------
// Updates the histogram image
// ---------------------------------------------------------
void HistogramView::updateHistogramImage( )
{
    /*
    size_t memSize = 4 * m_canvasRectangle.width( ) * m_canvasRectangle.height( );

    // Create the new image buffer
    m_imagebuffer = new unsigned char[memSize];
    memset( m_imagebuffer, 0, memSize );

    // Create the new image
    if( m_binMax != 0 )
    {
        switch(m_type)
        {
        case RenderController::VolumeHistogramType_Density:
            makeDensityHistogram( );
            break;

        case RenderController::VolumeHistogramType_Gradient1:
            break;

        case RenderController::VolumeHistogramType_Gradient2:
            break;
        }
    }

    // Convert the raw image for display
	m_histogramItem.setPixmap( QPixmap::fromImage( 
        QImage(m_imagebuffer, m_canvasRectangle.width( ), 
        m_canvasRectangle.height( ), m_canvasRectangle.width( )*4, 
        QImage::Format_ARGB32) ) );

    // Position the histogram image within the margins
    m_histogramItem.setOffset( m_canvasRectangle.left( ), 
        m_canvasRectangle.top( ) );
    */
}

// ---------------------------------------------------------
// Updates the histogram data buffer
// ---------------------------------------------------------
void HistogramView::updateHistogramData( )
{
    // Update the volume histogram bin data
    try
    {
        auto & renderer = MainWindow::instance->m_renderController;
        //m_bins = renderer.makeVolumeHistogram( m_type );
        //m_binMax = std::max( m_bins.front( ), m_bins.back( ) );
    }
    catch( vox::Error const& )
    {
        m_bins.clear( );
        m_binMax = 0;
    }

    // Redraw the histogram image
    //updateHistogramImage( );
}

// ---------------------------------------------------------
// Generates a density histogram volume data set
// ---------------------------------------------------------
void HistogramView::makeDensityHistogram( )
{
    auto const scale   = m_options & HistogramOptionF_LogScale;
    auto const max     = scale ? logf( 1.0f+m_binMax ) : m_binMax;
    auto const height  = m_canvasRectangle.height( );
    auto const width   = m_canvasRectangle.width( );
	auto const wscalar = height / width;
    auto const hscalar = height / max;

	// Draw histogram image
    if( scale )
    {
	    for( size_t i = 0; i < width; i++ )
	    {
		    size_t bucket = size_t(i * wscalar); // :TODO: Anti-aliasing

		    size_t maxHeight = height - size_t(logf(1.0f+m_bins[bucket]) * hscalar);

		    for( size_t j = maxHeight; j < height; j++ )
		    {
			    m_imagebuffer[size_t(i+j*width)*4+0] = 234;
			    m_imagebuffer[size_t(i+j*width)*4+1] = 217;
			    m_imagebuffer[size_t(i+j*width)*4+2] = 153;
			    m_imagebuffer[size_t(i+j*width)*4+3] = 160;
		    }
	    }
    }
    else
    {
	    for( size_t i = 0; i < width; i++ )
	    {
		    size_t bucket = size_t(i * wscalar); // :TODO: Anti-aliasing

		    size_t maxHeight = height - size_t(m_bins[bucket] * hscalar);

		    for( size_t j = maxHeight; j < height; j++ )
		    {
			    m_imagebuffer[size_t(i+j*width)*4+0] = 234;
			    m_imagebuffer[size_t(i+j*width)*4+1] = 217;
			    m_imagebuffer[size_t(i+j*width)*4+2] = 153;
			    m_imagebuffer[size_t(i+j*width)*4+3] = 160;
		    }
	    }
    }
}