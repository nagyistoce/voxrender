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

// Begin Definition
#ifndef SAMPLING_WIDGET_H
#define SAMPLING_WIDGET_H

// QT Includes
#include <QtGui/QWidget>

namespace Ui {
	class SamplingWidget;
}

// Volume data histogram widget
class SamplingWidget : public QWidget
{
	Q_OBJECT

public:
	explicit SamplingWidget( QWidget *parent = 0 );
	~SamplingWidget( );

	void Update( );
	void SetEnabled( bool enabled );

private:
	Ui::SamplingWidget *ui;

signals:
	void valuesChanged( );

private slots:
	void step1_changed( int value );
	void step1_changed( double value );
	void step2_changed( int value );
	void step2_changed( double value );
	void gradient_changed( int value );
	void gradient_changed( double value );
	void density_changed( int value );
	void density_changed( double value );

};

// End definition
#endif // SAMPLING_WIDGET_H

