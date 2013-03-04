/* ===========================================================================

	Project: HistogramWidget - Volume histogram widget

	Description:
	 Implements an interface for interactive viewing of volume histograms

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

// Include Headers
#include "ui_histogramwidget.h"
#include "histogramwidget.h"

// API Includes
#include "Core/VoxRender.h"

// Include Dependencies
#include "mainwindow.h"

// ---------------------------------------------------------
// Constuctor - Initialize widget slots and signals
// ---------------------------------------------------------
HistogramWidget::HistogramWidget( QWidget *parent ) : 
	QWidget(parent), ui(new Ui::HistogramWidget)
{
	ui->setupUi(this);
	
    // Create the histogram view 
	histogramView = new HistogramView( ui->frame_histogram );
	ui->histogramLayout->addWidget( histogramView, 0, 0, 1, 1 );

    // Setup the slider/spinBox signals
	connect( ui->slider_histogramGamma, SIGNAL(valueChanged(int)), 
		this, SLOT(gamma_changed(int)) );
	connect( ui->spinBox_histogramGamma, SIGNAL(valueChanged(double)), 
		this, SLOT(gamma_changed(double)) );
}
    
// ---------------------------------------------------------
// Destructor - Delete histogram view
// ---------------------------------------------------------
HistogramWidget::~HistogramWidget( )
{
    delete ui;
	delete histogramView;
}

// ---------------------------------------------------------
// Updates the histogram image
// ---------------------------------------------------------
void HistogramWidget::Update( )
{
	histogramView->updateHistogramImage( );
}

// ---------------------------------------------------------
// Toggle histogram log selection
// ---------------------------------------------------------
void HistogramWidget::on_checkBox_histogramLog_toggled( bool checked )
{
	histogramView->setLogEnabled( checked );
	histogramView->updateHistogramImage( );
}

// ---------------------------------------------------------
// Toggle histogram equalization selection
// ---------------------------------------------------------
void HistogramWidget::on_checkBox_histogramEqualize_toggled( bool checked )
{
	histogramView->setEqualizationEnabled( checked );
	histogramView->updateHistogramImage( );
}

// ---------------------------------------------------------
// Change histogram type selection
// ---------------------------------------------------------
void HistogramWidget::on_comboBox_histogramChannel_activated( QString str )
{
	if( str == "Density" )
	{
	}
	else if( str == "Gradient" )
	{
	}
	else if( str == "2nd Deriv." )
	{
	}
	else
    {
        // :TODO:
    }
}

// ---------------------------------------------------------
// Histogram gamma slider changed
// ---------------------------------------------------------
void HistogramWidget::gamma_changed( int value ) 
{ 
	gamma_changed
	( 
		(double)value / 
			( (double)ui->slider_histogramGamma->maximum( ) / 
				ui->spinBox_histogramGamma->maximum( ) ) 
	);
}

// ---------------------------------------------------------
// Histogram gamma slider changed
// ---------------------------------------------------------
void HistogramWidget::gamma_changed( double value ) 
{
	int sliderval = (int)(((double)ui->slider_histogramGamma->maximum( ) 
		/ ui->spinBox_histogramGamma->maximum( ) ) * value);

	ui->slider_histogramGamma->blockSignals( true );
	ui->spinBox_histogramGamma->blockSignals( true );
	ui->slider_histogramGamma->setValue( sliderval );
	ui->spinBox_histogramGamma->setValue( value );
	ui->slider_histogramGamma->blockSignals( false );
	ui->spinBox_histogramGamma->blockSignals( false );

	// Notify histogram view //
}