/* ===========================================================================

	Project: TransferItem - Transfer function display widget

	Description: Displays a transfer function mapping

    Copyright (C) 2013 Lucas Sherman

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
#include <QtWidgets/QGraphicsRectItem>
#include <QtWidgets/QGraphicsView>

// Include Dependencies
#include "nodeitem.h"
#include "edgeitem.h"

/** 
 * Transfer function display item 
 *
 * This class implements a QT Graphics Item which functions as a container for a 
 * representation of a transfer function along a single plane.
 *
 */
class TransferItem : public QObject, public QGraphicsRectItem
{
    Q_OBJECT 

public:
    /** Initializes the transfer function display */
	TransferItem(QGraphicsItem* parent = nullptr);

    /** Detaches the transfer function display from the scene object */
	~TransferItem() { }
    
    /** Recalculates relative positions */
    void onResizeEvent();

    void synchronizeView();

public slots:
    void mousePressEvent(QGraphicsSceneMouseEvent* pEvent);

private:
    std::list<std::shared_ptr<NodeItem>> m_nodes; ///< Graphics items for transfer nodes
    std::list<std::shared_ptr<EdgeItem>> m_edges; ///< Graphics items for transfer edges
};

// End definition
#endif // TRANSFER_ITEM_H