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

// Include Headers
#include "transferitem.h"

// Include Dependencies
#include "nodeitem.h"

// QT4 Dependencies
#include <QtGui/QGraphicsSceneMouseEvent>

// ---------------------------------------------------------
//  Constructor - Construct the transfer function editor
// ---------------------------------------------------------
TransferItem::TransferItem( QGraphicsItem *parent )
	: QGraphicsRectItem( parent )
{
    m_node = new NodeItem( this );
    m_node->setPos( QPointF( 10.0f, 10.0f ) );
}

// ---------------------------------------------------------
//  Regenerates the transfer function display nodes
// ---------------------------------------------------------
void TransferItem::update( )
{
}