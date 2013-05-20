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

// Begin definition
#ifndef HISTOGRAM_VIEW_H
#define HISTOGRAM_VIEW_H

// API Includes
#include "VoxLib/Core/VoxRender.h"

// Include Dependencies
#include "griditem.h"
#include "transferitem.h"

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

/** Implements a QT graphics view for volume histogram data and transfer functions */
class HistogramView : public QGraphicsView
{
	Q_OBJECT

public:
    /** Defines views for histograms and transfer functions */
    enum DataType
    {
        DataType_Begin,                   ///< Begin iterator for VolHistoType enumeration
        DataType_Density =DataType_Begin, ///< Density magnitude view of volume data
        DataType_DensityGrad,             ///< Density vs Gradient magnitude view of volume data
        DataType_DensityLap,              ///< Density vs Laplacian view of volume data
        DataType_End                      ///< End iterator for VolHistoType enumeration
    };

public:
    ~HistogramView();

    /** Constructs a new HistogramView with an option transfer function editor */
	HistogramView(QWidget *parent = 0, bool createTransferView = false);

    /** Updates the transfer function views */
    void updateTransfer();

    /** Enables log scaling of the density magnitude component of the histogram view */
	void setLogEnabled(bool enabled) 
    { 
        if( enabled ) m_options |= HistogramOptionF_LogScale;
		else m_options ^= HistogramOptionF_LogScale; 
    }

    /** Enables equalization of the histogram's generated results */
	void setEqualizationEnabled(bool enabled) 
    { 
        if( enabled ) m_options |= HistogramOptionF_Equalize;
		else m_options ^= HistogramOptionF_Equalize; 
    }
    
    /** Regenerates the histogram display image */
	void updateImage();

private slots:
    void updateHistogramData( );

private:
    // Histogram display option flags
    enum HistogramOptionF
    {
        HistogramOptionF_LogScale = 1<<0,
        HistogramOptionF_Equalize = 1<<1
    };
    
    // Generates a density histogram volume data set
    void makeDensityHistogram();

    // Updates the drawing canvas bounds
    void updateCanvas();

    std::vector<size_t> m_bins;
    size_t m_binMax;

    unsigned char* m_imagebuffer;
	unsigned int m_options;

	void wheelEvent(QWheelEvent *event);
	void resizeEvent(QResizeEvent *event);
	
	float zoomfactor;   ///< Current zoom level on histogram display
    
	TransferItem* m_transferItem; ///< Optional transfer function interaction item

    DataType m_type;

	QGraphicsScene      m_scene;            ///< Histogram view scene handle
	QGraphicsPixmapItem m_histogramItem;    ///< Histogram image
	GridItem            m_gridItem;         ///< Gridlines item object

    QRectF   m_canvasRectangle; ///< Full canvas for drawing operations
	QMargins m_margins;         ///< Margins for actual histogram display
};

// End definition
#endif // HISTOGRAM_VIEW_H