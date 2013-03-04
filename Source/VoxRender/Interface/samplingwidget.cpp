/* ===========================================================================

	Project: SamplingWidget - Sampling widget

	Description:
	 Implements an interface for modifying the sampling parameters of the
	 volume ray tracer.

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
#include "ui_samplingwidget.h"
#include "samplingwidget.h"

// Include Dependencies
#include "mainwindow.h"

// Constuctor - Initialize widget slots and signals
SamplingWidget::SamplingWidget( QWidget *parent ) : 
	QWidget(parent), ui(new Ui::SamplingWidget)
{
	ui->setupUi(this);

	// Connect slider / spinBox widgets to appropriate slots
	connect( ui->horizontalSlider_step1, SIGNAL(valueChanged(int)), 
		this, SLOT(step1_changed(int)) );
	connect( ui->doubleSpinBox_step1, SIGNAL(valueChanged(double)), 
		this, SLOT(step1_changed(double)) );
	connect( ui->horizontalSlider_step2, SIGNAL(valueChanged(int)), 
		this, SLOT(step2_changed(int)) );
	connect( ui->doubleSpinBox_step2, SIGNAL(valueChanged(double)), 
		this, SLOT(step2_changed(double)) );
	connect( ui->horizontalSlider_density, SIGNAL(valueChanged(int)), 
		this, SLOT(density_changed(int)) );
	connect( ui->doubleSpinBox_density, SIGNAL(valueChanged(double)), 
		this, SLOT(density_changed(double)) );
	connect( ui->horizontalSlider_gradient, SIGNAL(valueChanged(int)), 
		this, SLOT(gradient_changed(int)) );
	connect( ui->doubleSpinBox_gradient, SIGNAL(valueChanged(double)), 
		this, SLOT(gradient_changed(double)) );
}

// Destructor - Delete histogram view
SamplingWidget::~SamplingWidget( )
{
    delete ui;
}

// Primary step size slider changed
void SamplingWidget::step1_changed(int value)
{
	step1_changed
	( 
		(double)value / 
			( (double)ui->horizontalSlider_step1->maximum( ) / 
				ui->doubleSpinBox_step1->maximum( ) ) 
	);
}

// Primary step size spinBox changed
void SamplingWidget::step1_changed( double value )
{
	int sliderval = (int)(((double)ui->horizontalSlider_step1->maximum( ) 
		/ ui->doubleSpinBox_step1->maximum( ) ) * value);

	ui->horizontalSlider_step1->blockSignals( true );
	ui->doubleSpinBox_step1->blockSignals( true );
	ui->horizontalSlider_step1->setValue( sliderval );
	ui->doubleSpinBox_step1->setValue( value );
	ui->horizontalSlider_step1->blockSignals( false );
	ui->doubleSpinBox_step1->blockSignals( false );

	emit valuesChanged( );
}

// Secondary step size slider changed
void SamplingWidget::step2_changed(int value)
{
	step2_changed
	( 
		(double)value / 
			( (double)ui->horizontalSlider_step2->maximum( ) / 
				ui->doubleSpinBox_step2->maximum( ) ) 
	);
}

// Secondary step size spinBox changed
void SamplingWidget::step2_changed( double value )
{
	int sliderval = (int)(((double)ui->horizontalSlider_step2->maximum( ) 
		/ ui->doubleSpinBox_step2->maximum( ) ) * value);

	ui->horizontalSlider_step2->blockSignals( true );
	ui->doubleSpinBox_step2->blockSignals( true );
	ui->horizontalSlider_step2->setValue( sliderval );
	ui->doubleSpinBox_step2->setValue( value );
	ui->horizontalSlider_step2->blockSignals( false );
	ui->doubleSpinBox_step2->blockSignals( false );

	emit valuesChanged( );
}

// Density scale slider changed
void SamplingWidget::density_changed(int value)
{
	density_changed
	( 
		(double)value / 
			( (double)ui->horizontalSlider_density->maximum( ) / 
				ui->doubleSpinBox_density->maximum( ) ) 
	);
}

// Density scale spinBox changed
void SamplingWidget::density_changed( double value )
{
	int sliderval = (int)(((double)ui->horizontalSlider_density->maximum( ) 
		/ ui->doubleSpinBox_density->maximum( ) ) * value);

	ui->horizontalSlider_density->blockSignals( true );
	ui->doubleSpinBox_density->blockSignals( true );
	ui->horizontalSlider_density->setValue( sliderval );
	ui->doubleSpinBox_density->setValue( value );
	ui->horizontalSlider_density->blockSignals( false );
	ui->doubleSpinBox_density->blockSignals( false );

	emit valuesChanged( );
}

// Gradient cutoff slider changed
void SamplingWidget::gradient_changed(int value)
{
	gradient_changed
	( 
		(double)value / 
			( (double)ui->horizontalSlider_gradient->maximum( ) / 
				ui->doubleSpinBox_gradient->maximum( ) ) 
	);
}

// Gradient cutoff spinBox changed
void SamplingWidget::gradient_changed( double value )
{
	int sliderval = (int)(((double)ui->horizontalSlider_gradient->maximum( ) 
		/ ui->doubleSpinBox_gradient->maximum( ) ) * value);

	ui->horizontalSlider_gradient->blockSignals( true );
	ui->doubleSpinBox_gradient->blockSignals( true );
	ui->horizontalSlider_gradient->setValue( sliderval );
	ui->doubleSpinBox_gradient->setValue( value );
	ui->horizontalSlider_gradient->blockSignals( false );
	ui->doubleSpinBox_gradient->blockSignals( false );

	emit valuesChanged( );
}