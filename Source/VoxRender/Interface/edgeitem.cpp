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

// Include Header
#include "edgeitem.h"

// Include Dependencies
#include "nodeitem.h"

// --------------------------------------------------------------------
//  Static member initialization
// --------------------------------------------------------------------
QPen EdgeItem::m_penNormal     = QPen(QBrush(QColor::fromHsl(0, 100, 120)), 1.0);
QPen EdgeItem::m_penHighlight  = QPen(QBrush(QColor::fromHsl(0, 100, 120)), 1.0);
QPen EdgeItem::m_penDisabled   = QPen(QBrush(QColor::fromHsl(0, 0, 180)), 1.0);

// --------------------------------------------------------------------
//  Constructor - Initializes the edge items node endpoints
// --------------------------------------------------------------------
EdgeItem::EdgeItem(QGraphicsItem* parent, std::weak_ptr<NodeItem> node1, std::weak_ptr<NodeItem> node2) :
	QGraphicsLineItem(parent),
    m_node1(node1),
    m_node2(node2)
{
}

// --------------------------------------------------------------------
//  Draws the edge between the nodeitems using the proper pen
// --------------------------------------------------------------------
void EdgeItem::paint(QPainter* pPainter, const QStyleOptionGraphicsItem* pOption, QWidget* pWidget)
{
    // Set the pen 
	if (isEnabled())
	{
		if (isUnderMouse() || isSelected()) 
            setPen(m_penHighlight);
		else 
            setPen(m_penNormal);
	}
	else
	{
		setPen(m_penDisabled);
	}

    // Set the endpoints
    auto node1 = m_node1.lock();
    auto node2 = m_node2.lock();
    if (node1 && node2)
    {
        setLine(node1->x(), node1->y(), 
                node2->x(), node2->y());
    }

    // Draw the line
	QGraphicsLineItem::paint(pPainter, pOption, pWidget);
}