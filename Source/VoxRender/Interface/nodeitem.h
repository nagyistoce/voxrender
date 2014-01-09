/* ===========================================================================

	Project: VoxRender - Transfer function node object for GraphicsView

	Defines a class for managing buffers on devices using CUDA.

	Lucas Sherman, email: LucasASherman@gmail.com

    MODIFIED FROM EXPOSURE RENDER'S "nodeitem.h" SOURCE FILE:

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
#ifndef NODE_ITEM_H
#define NODE_ITEM_H

// QT4 Dependencies
#include <QtWidgets/QGraphicsView>
#include <QtWidgets/QGraphicsEllipseItem>
#include <QtWidgets/QGraphicsScene>

// Include Dependencies
#include <memory>

// Parent graphics item class
class TransferItem;

/** Graphics item representing a POI on the transfer function */
class NodeItem : public QGraphicsEllipseItem
{
public:
    /** Constructs a new NodeItem for the specified transfer data */
	NodeItem(TransferItem* parent, float x, float y, std::shared_ptr<void> data = nullptr);

    /** Updates the position of the node item */
    void setPosition(float x, float y);
    
    /** Returns the transfer function data associated with this element */
    std::shared_ptr<void> data() { return m_data; }

protected:
	virtual void mousePressEvent(QGraphicsSceneMouseEvent* pEvent);
	virtual void mouseReleaseEvent(QGraphicsSceneMouseEvent* pEvent);

	virtual void paint( 
        QPainter* pPainter, 
		const QStyleOptionGraphicsItem* pOption, 
		QWidget* pWidget 
        );

    virtual QVariant itemChange(GraphicsItemChange Change, const QVariant& Value);

private:
    TransferItem * m_parent; ///< Handle to parent 

    std::shared_ptr<void> m_data; ///< Associated data

    bool m_ignorePosChange; ///< Indicates lock for synchronizing node attributes
};

// End definition
#endif // NODE_ITEM_H