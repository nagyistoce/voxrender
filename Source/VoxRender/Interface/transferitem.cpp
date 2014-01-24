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

// Transfer function modification wrapper for auto-update
#define DO_LOCK(T, X)   \
    T->lock();          \
    X                   \
    T->setDirty();      \
    T->unlock();
// :TODO: Check auto update set 

using namespace vox;

namespace {
namespace filescope {

    /** Wrapper class for identifying node items on a quad */
    class QuadIndex 
    {
    public:
        QuadIndex(std::shared_ptr<Quad> q, Quad::Node n) :
            quad(q), node(n)
        {
        }

        std::shared_ptr<Quad> quad;
        Quad::Node node;
    };

} // namespace filescope
} // namespace anonymous

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

    if (auto transfer1D = dynamic_cast<Transfer1D*>(transfer.get()))
    {
        // Update transfer function nodes
        auto nodes = transfer1D->nodes();
        std::shared_ptr<NodeItem> prevItem;
        BOOST_FOREACH (auto & node, nodes)
        {
            // Create a graphics object for the next transfer function node
            auto nodeItem = std::shared_ptr<NodeItem>(new NodeItem(
                this, node->density, node->material->opticalThickness, node));
            m_nodes.push_back(nodeItem);

            // Create an edge between the previous and next nodes
            if (prevItem)
            {
                auto edgeItem = std::shared_ptr<EdgeItem>( 
                    new EdgeItem(this, prevItem, nodeItem));
                m_edges.push_back(edgeItem);
            }

            // Set the previous node
            prevItem = nodeItem;
        }
    }
    else if (auto transfer2D = dynamic_cast<Transfer2D*>(transfer.get()))
    {
        auto quads = transfer2D->quads();
        BOOST_FOREACH (auto & quad, quads)
        {
            auto ll = std::shared_ptr<NodeItem>(new NodeItem(this, 
                quad->position[0] - quad->widths[1] / 2,
                quad->position[1] - quad->heights[0] / 2,
                std::make_shared<filescope::QuadIndex>(quad, Quad::Node_LL)
                ));
            auto ul = std::shared_ptr<NodeItem>(new NodeItem(this, 
                quad->position[0] - quad->widths[0] / 2,
                quad->position[1] + quad->heights[0] / 2,
                std::make_shared<filescope::QuadIndex>(quad, Quad::Node_UL)
                ));
            auto lr = std::shared_ptr<NodeItem>(new NodeItem(this, 
                quad->position[0] + quad->widths[1] / 2,
                quad->position[1] - quad->heights[1] / 2,
                std::make_shared<filescope::QuadIndex>(quad, Quad::Node_LR)
                ));
            auto ur = std::shared_ptr<NodeItem>(new NodeItem(this, 
                quad->position[0] + quad->widths[0] / 2,
                quad->position[1] + quad->heights[1] / 2,
                std::make_shared<filescope::QuadIndex>(quad, Quad::Node_UR)
                ));
            auto c = std::shared_ptr<NodeItem>(new NodeItem(this, 
                quad->position[0],
                quad->position[1],
                std::make_shared<filescope::QuadIndex>(quad, Quad::Node_End)
                ));
            m_edges.push_back(std::shared_ptr<EdgeItem>(new EdgeItem(this, ll, lr)));
            m_edges.push_back(std::shared_ptr<EdgeItem>(new EdgeItem(this, ul, ur)));
            m_edges.push_back(std::shared_ptr<EdgeItem>(new EdgeItem(this, ll, ul)));
            m_edges.push_back(std::shared_ptr<EdgeItem>(new EdgeItem(this, lr, ur)));
            m_nodes.push_back(ll);
            m_nodes.push_back(ul);
            m_nodes.push_back(lr);
            m_nodes.push_back(ur);
            m_nodes.push_back(c);
            // c->color = green, radius = less :TODO:
        }
    }
}

// ----------------------------------------------------------------------------
//  Updates the specified node on the transfer function
// ----------------------------------------------------------------------------
void TransferItem::updateNode(std::shared_ptr<vox::Node> node)
{
    BOOST_FOREACH (auto & nodeElem, m_nodes)
    if (nodeElem->data() == node) 
    {
        nodeElem->setSelected(node == MainWindow::instance->transferWidget()->selectedNode());
        nodeElem->setPosition(node->density, node->material->opticalThickness);
    }
}

// ----------------------------------------------------------------------------
//  Updates the specified quad on the transfer function
// ----------------------------------------------------------------------------
void TransferItem::updateQuad(std::shared_ptr<vox::Quad> quad)
{
    BOOST_FOREACH (auto & nodeElem, m_nodes)
    {
        auto quadItem = std::static_pointer_cast<filescope::QuadIndex>(nodeElem->data());
        if (quadItem->quad == quad)
        switch (quadItem->node)
        {
        case Quad::Node_LL:
            nodeElem->setPosition(
                quad->position[0] - quad->widths[1] / 2,
                quad->position[1] - quad->heights[0] / 2);
            break;
        case Quad::Node_UL:
            nodeElem->setPosition(
                quad->position[0] - quad->widths[0] / 2,
                quad->position[1] + quad->heights[0] / 2);
            break;
        case Quad::Node_LR:
            nodeElem->setPosition(
                quad->position[0] + quad->widths[1] / 2,
                quad->position[1] - quad->heights[1] / 2);
            break;
        case Quad::Node_UR:
            nodeElem->setPosition(
                quad->position[0] + quad->widths[0] / 2,
                quad->position[1] + quad->heights[1] / 2);
            break;
        case Quad::Node_End:
            nodeElem->setPosition(
                quad->position[0],
                quad->position[1]);
            break;
        }
    }
}

// ----------------------------------------------------------------------------
//  Synchronizes nodeItem changes with the associated transfer function feature
// ----------------------------------------------------------------------------
void TransferItem::onNodeItemChanged(NodeItem * item, float x, float y)
{
    auto transfer = MainWindow::instance->scene().transfer;
    if (transfer->type() == Transfer1D::typeID())
    {
        auto node = std::static_pointer_cast<vox::Node>(item->data());
        DO_LOCK(transfer, node->density = x; node->material->opticalThickness = y;)
        MainWindow::instance->transferWidget()->setSelectedNode(node);
    }
    else if (transfer->type() == Transfer2D::typeID())
    {
        transfer->lock();

        auto quadItem = std::static_pointer_cast<filescope::QuadIndex>(item->data());
        switch (quadItem->node)
        {
        case Quad::Node_LL:
            quadItem->quad->widths[1]  = (quadItem->quad->position[0] - x)*2;
            quadItem->quad->heights[0] = (quadItem->quad->position[0] - y)*2; break;
        case Quad::Node_LR:
            quadItem->quad->widths[1]  = (x - quadItem->quad->position[0])*2;
            quadItem->quad->heights[1] = (quadItem->quad->position[0] - y)*2; break;
        case Quad::Node_UL:
            quadItem->quad->widths[0]  = (quadItem->quad->position[0] - x)*2;
            quadItem->quad->heights[0] = (y - quadItem->quad->position[0])*2; break;
        case Quad::Node_UR:
            quadItem->quad->widths[0]  = (x - quadItem->quad->position[0])*2;
            quadItem->quad->heights[1] = (y - quadItem->quad->position[0])*2; break;
        case Quad::Node_End: 
            quadItem->quad->position[0] = x;
            quadItem->quad->position[1] = y;
            break;
        }

        transfer->unlock();
        transfer->setDirty(true);

        updateQuad(quadItem->quad);
    }
}

// ----------------------------------------------------------------------------
//  Ensures a node item does not stray outside its bounds
// ----------------------------------------------------------------------------
void TransferItem::onNodeItemChange(NodeItem * item, QPointF & pos)
{
    auto transfer = MainWindow::instance->scene().transfer;
    if (transfer->type() == Transfer1D::typeID())
    {
        for (auto iter = m_nodes.begin(); iter != m_nodes.end(); ++iter)
        if ((*iter).get() == item)
        {
            if (iter != m_nodes.begin()) 
            {
                auto pX = (*--iter)->pos().x();
                if (pX > pos.x()) pos.setX(pX);
                ++iter;
            }
        
            ++iter;

            if (iter != m_nodes.end())
            {
                auto nX = (*iter)->pos().x();
                if (nX < pos.x()) pos.setX(nX);
            }

            return;
        }
    } 
    else if (transfer->type() == Transfer2D::typeID())
    {
        auto quadItem = std::static_pointer_cast<filescope::QuadIndex>(item->data());
        auto & quad = quadItem->quad;
            
        auto sw = rect().width();
        auto sh = rect().height();
        auto w = high(quad->widths[0],  quad->widths[1])  / 2 * sw;
        auto h = high(quad->heights[0], quad->heights[1]) / 2 * sh;
        auto x = rect().left();
        auto y = rect().top();

        switch (quadItem->node)
        {
        //case Quad::Node_LL:
        //    if (pos.x() > x+sw*quad->position[0]) pos.setX(x+sw*quad->position[0]);
        //    if (pos.y() < y+sh*quad->position[1]) pos.setY(y+sh*quad->position[1]);
        //    break;
        //case Quad::Node_LR:
        //    if (pos.x() > x+sw*quad->position[0]) pos.setX(x+sw*quad->position[0]);
        //    if (pos.y() > y+sh*quad->position[1]) pos.setY(y+sh*quad->position[1]);
        //    break;
        //case Quad::Node_UL:
        //    if (pos.x() < x+sw*quad->position[0]) pos.setX(x+sw*quad->position[0]);
        //    if (pos.y() < y+sh*quad->position[1]) pos.setY(y+sh*quad->position[1]);
        //    break;
        //case Quad::Node_UR:
        //    if (pos.x() < x+sw*quad->position[0]) pos.setX(x+sw*quad->position[0]);
        //    if (pos.y() > y+sh*quad->position[1]) pos.setY(y+sh*quad->position[1]);
        //    break;
        case Quad::Node_End: 
            if (pos.x() < x+w) pos.setX(x+w);
            if (pos.y() < y+h) pos.setY(y+h);
            if (pos.x() > x+sw-w) pos.setX(x+sw-w);
            if (pos.y() > y+sh-h) pos.setY(y+sh-h);
            break;
        }
    }
}

// ----------------------------------------------------------------------------
//  Updates the selected item in the transfer function editor
// ----------------------------------------------------------------------------
void TransferItem::onNodeItemSelected(NodeItem * item, bool selected)
{
    auto transfer = MainWindow::instance->scene().transfer;
    if (transfer->type() == Transfer1D::typeID())
    {
        auto node = std::static_pointer_cast<vox::Node>(item->data());
        if (!selected) MainWindow::instance->setTransferNode(node);
        else MainWindow::instance->setTransferNode(nullptr);
    }
    else if (transfer->type() == Transfer2D::typeID())
    {
        auto quadItem = std::static_pointer_cast<filescope::QuadIndex>(item->data());
        if (!selected) MainWindow::instance->setTransferQuad(quadItem->quad, quadItem->node);
        else ;
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

    if (auto transfer1D = dynamic_cast<Transfer1D*>(transfer.get()))
    {
        auto pos = pEvent->pos();
        auto ext = rect();
        auto density = (pos.x()-ext.left()) / ext.width();
        auto thickness = (pos.y()-ext.top()) / ext.height();
        auto node = Node::create(density);
        node->material->opticalThickness = 1.0f - thickness;
        transfer1D->addNode(node);

        MainWindow::instance->setTransferNode(node);
    }
    else if (auto transfer2D = dynamic_cast<Transfer2D*>(transfer.get()))
    {
        auto clickPos = pEvent->pos();
        auto ext = rect();
        auto density = (clickPos.x()-ext.left()) / ext.width();
        auto gradient = (clickPos.y()-ext.top())  / ext.height();
        auto quad = Quad::create();
        quad->position[0] = density;
        quad->position[1] = 1.f - gradient;
        transfer2D->addQuad(quad);

        auto pos = pEvent->pos();

        MainWindow::instance->setTransferNode(nullptr);
    }

    MainWindow::instance->transferWidget()->onTransferFunctionChanged();
}

// ----------------------------------------------------------------------------
//  Recalculates the transfer node positions following a scene rect resize
// ----------------------------------------------------------------------------
void TransferItem::onResizeEvent()
{
    auto transfer = MainWindow::instance->scene().transfer;
    if (!transfer) return;

    if (transfer->type() == Transfer1D::typeID())
    {
        BOOST_FOREACH(auto & item, m_nodes)
        {
            auto node = std::static_pointer_cast<vox::Node>(item->data());
            float x = node->density;
            float y = node->material->opticalThickness;
            item->setPosition(x, y);
        }
    }
    else if (transfer->type() == Transfer2D::typeID())
    {
        synchronizeView();
    }
}