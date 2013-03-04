/* ===========================================================================

	Project: TransferItem - Transfer function display widget

	Description:
	 Displays an n-dimensional set of transfer function nodes.

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
#ifndef TRANSFER_ITEM_H
#define TRANSFER_ITEM_H

// QT4 Dependencies
#include <QtGui/QGraphicsRectItem>
#include <QtGui/QGraphicsView>

class NodeItem;

/** 
 * Transfer function display item 
 *
 * This class implements a QT Graphics Item which functions as a container for a 
 * representation of a transfer function along a single plane.
 *
 */
class TransferItem : public QGraphicsRectItem
{
public:
    /** Initializes the transfer function display */
	TransferItem(QGraphicsItem* parent = nullptr);

    /** Detaches the transfer function display from the scene object */
	~TransferItem() { }

private:
    void update( );

    NodeItem* m_node;
};

// End definition
#endif // TRANSFER_ITEM_H