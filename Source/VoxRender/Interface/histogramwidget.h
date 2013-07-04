/* ===========================================================================

	Project: HistogramView - Volume histogram widget

	Based on luxrender histogram widget class.
	Lux Renderer website : http://www.luxrender.net 

	Description:
	 Implements an interface for run-time logging and error handling

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
#ifndef HISTOGRAM_WIDGET_H
#define HISTOGRAM_WIDGET_H

// QT4 Dependencies
#include <QtWidgets/QWidget>

// Include dependencies
#include "histogramview.h"

namespace Ui
{
	class HistogramWidget;
}

// Volume data histogram widget
class HistogramWidget : public QWidget
{
	Q_OBJECT

public:

	HistogramWidget( QWidget *parent = 0 );
	~HistogramWidget( );

	void Update( );
	void SetEnabled( bool enabled );

private:
	Ui::HistogramWidget * ui;
	HistogramView * histogramView;

private slots:
	void on_comboBox_histogramChannel_activated(QString str);
};

// End definition
#endif // HISTOGRAMWIDGET_H