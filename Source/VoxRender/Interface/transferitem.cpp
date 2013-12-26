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
    disconnect(this, SLOT(updateNode(std::shared_ptr<vox::Node>)));
    connect(MainWindow::instance->transferWidget(), SIGNAL(nodePositionChanged(std::shared_ptr<vox::Node>)), 
            this, SLOT(updateNode(std::shared_ptr<vox::Node>)));

    m_nodes.clear();
    m_edges.clear();

    auto transfer = MainWindow::instance->scene().transfer;
    if (!transfer) return;

    if (true)
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
    }
}

// ----------------------------------------------------------------------------
//  Updates the specified node on the transfer function
// ----------------------------------------------------------------------------
void TransferItem::updateNode(std::shared_ptr<vox::Node> node)
{
    BOOST_FOREACH (auto & nodeElem, m_nodes)
    if (nodeElem->node() == node) 
    {
        nodeElem->setSelected(node == MainWindow::instance->transferWidget()->selectedNode());
        nodeElem->updatePosition();
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

    auto node = Node::create();

    auto pos = pEvent->pos();
    auto ext = rect();
    node->setPosition(0, (pos.x()-ext.left()) / ext.width());
    node->material()->setOpticalThickness(1.0f - (pos.y()-ext.top()) / ext.height());

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