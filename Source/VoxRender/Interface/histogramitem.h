/* ===========================================================================

	Project: HistogramItem - Histogram display

	Description:
	 Encapsulates histogram images generated for the volume data set.

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
#ifndef HISTOGRAM_ITEM_H
#define HISTOGRAM_ITEM_H

// QT Includes
#include <QtGui/QGraphicsRectItem>
#include <QtGui/QWidget>

// Volume data histogram widget
class HistogramItem : public QGraphicsRectItem
{
public:

	HistogramItem( QWidget *parent = 0 );
	~HistogramItem( );

private:

private slots:
};

// End definition
#endif // HISTOGRAM_ITEM_H