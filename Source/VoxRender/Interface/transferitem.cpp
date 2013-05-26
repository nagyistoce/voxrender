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

// Include Headers
#include "transferitem.h"

// Include Dependencies
#include "mainwindow.h"

// VoxLib Dependencies
#include "VoxLib/Scene/Transfer.h"

// QT Dependencies
#include <QtWidgets/QGraphicsSceneMouseEvent>

using namespace vox;

// ----------------------------------------------------------------------------
//  Constructor - Construct the transfer function editor
// ----------------------------------------------------------------------------
TransferItem::TransferItem(QGraphicsItem* parent)
	: QGraphicsRectItem(parent)
{
}

// ----------------------------------------------------------------------------
//  Regenerates the transfer function display nodes
// ----------------------------------------------------------------------------
void TransferItem::synchronizeView()
{
    m_nodes.clear();
    m_edges.clear();

    if (auto transfer = MainWindow::instance->activeScene.transfer)
    {
        // Update transfer function nodes
        auto nodes = transfer->nodes();
        std::shared_ptr<NodeItem> prevItem;
        BOOST_FOREACH(auto & node, nodes)
        {
            // Create a graphics object for the next transfer function node
            auto nodeItem = std::shared_ptr<NodeItem>( new NodeItem(this, node) );
            nodeItem->setZValue(500);
            m_nodes.push_back( nodeItem );

            // Create an edge between the previous and next nodes
            if (prevItem)
            {
                auto edgeItem = std::shared_ptr<EdgeItem>( 
                    new EdgeItem(this, prevItem, nodeItem) );
                edgeItem->setZValue(400);
                m_edges.push_back( edgeItem );
            }

            // Set the previous node
            prevItem = nodeItem;
        }

        // Update transfer function edges
    }
}

// ----------------------------------------------------------------------------
//  Deselects or creates new nodes within the transfer function on right click 
//  events
// ----------------------------------------------------------------------------
void TransferItem::mousePressEvent(QGraphicsSceneMouseEvent* pEvent)
{
    QGraphicsRectItem::mousePressEvent(pEvent);

    if (pEvent->button() == Qt::LeftButton) return;

    auto transfer = MainWindow::instance->scene().transfer;

    auto node = std::make_shared<Node>();

    transfer->addNode(node);

    MainWindow::instance->setTransferNode(node);

    MainWindow::instance->transferWidget()->onTransferFunctionChanged();
}

// ----------------------------------------------------------------------------
//  Recalculates the transfer node positions following a scene rect resize
// ----------------------------------------------------------------------------
void TransferItem::onResizeEvent()
{
    BOOST_FOREACH(auto & node, m_nodes)
    {
        node->updatePosition();
    }
}