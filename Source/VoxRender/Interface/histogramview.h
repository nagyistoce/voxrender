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
#include <QtGui/QGraphicsView>
#include <QtGui/QGraphicsScene>
#include <QtGui/QApplication>
#include <QtCore/QEvent>
#include <QtGui/QGraphicsPixmapItem>
#include <QtGui/QWheelEvent>
#include <QtGui/QMouseEvent>
#include <QtGui/QMatrix>
#include <QtCore/QPoint>
#include <QtGui/QClipboard>
#include <QtCore/QMargins>

// Histogram graphics view
class HistogramView : public QGraphicsView
{
	Q_OBJECT

public:
	HistogramView( QWidget *parent = 0 );
    ~HistogramView( );

    // Enables log scaling of the histogram data
	void setLogEnabled( bool enabled ) { 
        if( enabled ) m_options |= HistogramOptionF_LogScale;
		else m_options ^= HistogramOptionF_LogScale; }

    // Enables equalization of the histogram data
	void setEqualizationEnabled( bool enabled ) { 
        if( enabled ) m_options |= HistogramOptionF_Equalize;
		else m_options ^= HistogramOptionF_Equalize; }
    
    // Updates the histogram image
	void updateHistogramImage( );

private slots:
    void updateHistogramData( );

private:
    // Histogram display option flag
    enum HistogramOptionF
    {
        HistogramOptionF_LogScale = 1<<0,
        HistogramOptionF_Equalize = 1<<1
    };
    
    // Generates a density histogram volume data set
    void HistogramView::makeDensityHistogram( );

    //vox::RenderController::VolumeHistogramType m_type;
    std::vector<size_t> m_bins;
    size_t m_binMax;

    unsigned char* m_imagebuffer;
	unsigned int m_options;

	void wheelEvent( QWheelEvent *event );
	void resizeEvent( QResizeEvent *event );
	
	float zoomfactor;

	QGraphicsScene m_scene;
	TransferItem m_transferItem;
	QGraphicsPixmapItem m_histogramItem;
	GridItem m_gridItem;

    QRectF m_canvasRectangle;
	QMargins m_margins;
};

// End definition
#endif // HISTOGRAM_VIEW_H