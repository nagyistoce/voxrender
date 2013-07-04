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
#include "VoxLib/Core/Logging.h"
#include "VoxLib/Core/Format.h"

// Include Dependencies
#include "mainwindow.h"

using namespace vox;

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
	//histogramView->updateImage( );
}

// ---------------------------------------------------------
// Change histogram type selection
// ---------------------------------------------------------
void HistogramWidget::on_comboBox_histogramChannel_activated(QString str)
{
	if (str == "Density")
	{
        histogramView->setType(HistogramView::DataType_Density);
	}
	else if (str == "Gradient")
	{
        histogramView->setType(HistogramView::DataType_DensityGrad);
	}
	else if (str == "Laplacian")
	{
        histogramView->setType(HistogramView::DataType_DensityLap);
	}
	else
    {
        VOX_LOG_ERROR(Error_Bug, "GUI", format("Unrecognized histogram type: %1%", str.toLatin1().data()));
    }
}