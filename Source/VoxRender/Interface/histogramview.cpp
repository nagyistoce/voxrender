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

namespace {
namespace filescope {

    // --------------------------------------------------------------------
    // :TODO:
    // --------------------------------------------------------------------
    template<typename T> Vector2f maxValueRange(size_t elements, UInt8 const* raw)
    {
        Vector<T,2> minMax(std::numeric_limits<T>::max(), static_cast<T>(0));

        T const* data = reinterpret_cast<T const*>(raw);

        for (size_t i = 0; i < elements; i++)
        {
            if (minMax[0] > *data) minMax[0] = *data;
            else if (minMax[1] < *data) minMax[1] = *data;

            data++;
        }

        Vector2f result = static_cast<Vector2f>(minMax) / 
            static_cast<float>(std::numeric_limits<T>::max());

        return result;
    }
    
    // --------------------------------------------------------------------
    // :TODO:
    // --------------------------------------------------------------------
    template<typename T> std::vector<size_t> generateHistogramBins(size_t nBins, size_t elements, UInt8 const* raw)
    {
        Vector2f range = maxValueRange<T>(elements, raw);

        std::vector<size_t> bins(nBins, 0);

        T const* data = reinterpret_cast<T const*>(raw);
        float    max  = static_cast<float>(std::numeric_limits<T>::max());

        for (size_t i = 0; i < elements; i++)
        {
            float  sample           = static_cast<float>(data[i]) / max;
            float  normalizedSample = (sample - range[0]) / (range[1] - range[0]);

            size_t bin = clamp<size_t>(static_cast<size_t>(normalizedSample*nBins), 0, nBins-1);

            bins[bin]++;
        }

        return bins;
    }

    // ----------------------------------------------------------------------------
    //  Generates histogram information for the volume
    // ----------------------------------------------------------------------------
    std::vector<size_t> generateHistogram(size_t nBins, std::shared_ptr<Volume> volume)
    {
        size_t elements   = volume->extent().fold<size_t>(1, &mul);
        UInt8 const* data = volume->data();

        switch (volume->type())
        {
            case Volume::Type_UInt8:  return filescope::generateHistogramBins<UInt8>(nBins, elements, data);
            case Volume::Type_UInt16: return filescope::generateHistogramBins<UInt16>(nBins, elements, data);
            default:
                throw Error(__FILE__, __LINE__, VSR_LOG_CATEGORY,
                    format("Unsupported volume data type (%1%)", 
                           Volume::typeToString(volume->type())),
                    Error_NotImplemented);
        }
    }

} // namespace filescope
} // namespace anonymous

// ---------------------------------------------------------
// Constructor - Initialize display scene and view parameters
// ---------------------------------------------------------
HistogramView::HistogramView(QWidget *parent, bool createTransferView) : 
    QGraphicsView(parent),
    m_type(DataType_Density),
	m_margins(30, 20, -20, -30),
    m_imagebuffer(nullptr), 
    m_transferItem(nullptr),
    m_histogramItem(nullptr),
    m_gridItem(nullptr),
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

	updateCanvas(); // Force initial update of scene
}
    
// ---------------------------------------------------------
// Destructor - Frees the image buffer
// ---------------------------------------------------------
HistogramView::~HistogramView()
{
    delete m_imagebuffer;
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

    // Regenerate the background image
    updateImage();

	QGraphicsView::resizeEvent(event);
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
//  
// ---------------------------------------------------------
void HistogramView::updateCanvas()
{
}

// ---------------------------------------------------------
// Updates the histogram image
// ---------------------------------------------------------
void HistogramView::updateImage( )
{
    size_t memSize = 4 * m_canvasRectangle.width( ) * m_canvasRectangle.height( );

    // Create the new image buffer
    m_imagebuffer = new unsigned char[memSize];
    memset( m_imagebuffer, 0, memSize );

    // Create the new image
    if( m_binMax != 0 )
    {
        makeDensityHistogram( );
    }

    // Convert the raw image for display
	m_histogramItem.setPixmap( QPixmap::fromImage( 
        QImage(m_imagebuffer, m_canvasRectangle.width( ), 
        m_canvasRectangle.height( ), m_canvasRectangle.width( )*4, 
        QImage::Format_ARGB32) ) );

    // Position the histogram image within the margins
    m_histogramItem.setOffset( m_canvasRectangle.left( ), 
        m_canvasRectangle.top( ) );
}

// ---------------------------------------------------------
// Updates the histogram data buffer
// ---------------------------------------------------------
void HistogramView::updateHistogramData()
{
    // Update the volume histogram bin data
    try
    {
        auto & renderer = MainWindow::instance->m_renderController;

        m_bins = filescope::generateHistogram(256, MainWindow::instance->scene().volume);

        m_binMax = std::max(m_bins.front(), m_bins.back());
    }
    catch( vox::Error const& )
    {
        m_bins.clear( );
        m_binMax = 0;
    }

    // Redraw the histogram image
    //updateImage( );
}

// ---------------------------------------------------------
//  Generates a density histogram volume data set
// ---------------------------------------------------------
void HistogramView::makeDensityHistogram()
{
    auto const scale   = m_options & HistogramOptionF_LogScale;
    auto const max     = scale ? logf(1.0f+m_binMax) : m_binMax;
    auto const height  = m_canvasRectangle.height();
    auto const width   = m_canvasRectangle.width();
	auto const wscalar = m_bins.size() / width;
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