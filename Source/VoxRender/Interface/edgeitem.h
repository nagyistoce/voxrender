/* ===========================================================================

	Project: VoxRender - Transfer function edge object for GraphicsView

	Defines a class for drawing lines between transfer nodes

	Lucas Sherman, email: LucasASherman@gmail.com

    MODIFIED FROM EXPOSURE RENDER'S "nodeitem.cpp" SOURCE FILE:

    Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
    All rights reserved.

    Redistribution and use in source and binary forms, with or without 
    modification, are permitted provided that the following conditions are 
    met:

      - Redistributions of source code must retain the above copyright 
        notice, this list of conditions and the following disclaimer.
      - Redistributions in binary form must reproduce the above copyright 
        notice, this list of conditions and the following disclaimer in the 
        documentation and/or other materials provided with the distribution.
      - Neither the name of the <ORGANIZATION> nor the names of its 
        contributors may be used to endorse or promote products derived from 
        this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED 
    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
    PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR 
    CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
    EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; 
    OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
    WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR 
    OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, 
    EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

=========================================================================== */

// Begin definition
#ifndef EDGE_ITEM_H
#define EDGE_ITEM_H

// Include Dependencies
#include <QtGui/QGraphicsLineItem>
#include <QtGui/QPen>

class NodeItem;

/** Edge item for drawing connections between transfer function nodes */
class EdgeItem : public QGraphicsLineItem
{
public:
    /** Initializes an edge between two nodes */
	EdgeItem(QGraphicsItem* parent, std::weak_ptr<NodeItem> node1, std::weak_ptr<NodeItem> node2);

    /** Draws a line between the edgeitem's nodes */
	void paint(QPainter* pPainter, const QStyleOptionGraphicsItem* pOption, QWidget* pWidget);

    /** Sets the normal (unselected) color of the line */
    static void setPenNormal(QPen const& pen) { m_penNormal = pen; }

    /** Sets the highlighted (selected) color of the line */
    static void setPenHighlight(QPen const& pen) { m_penHighlight = pen; }

    /** Sets the disabled color of the line */
    static void setPenDisabled(QPen const& pen) { m_penDisabled = pen; }

private:
    std::weak_ptr<NodeItem> m_node1; ///< Reference to an endpoint
    std::weak_ptr<NodeItem> m_node2; ///< Reference to an endpoint

	static QPen	m_penNormal;
	static QPen m_penHighlight;
	static QPen m_penDisabled;
};

// End definition
#endif // EDGE_ITEM_H