/* ===========================================================================

	Project: VoxRender - Transfer function node object for GraphicsView

	Defines a class for managing buffers on devices using CUDA.

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
#include "nodeitem.h"

// VoxRender Includes
#include "VoxLib/Core/Format.h"
#include "VoxLib/Core/Logging.h"
#include "VoxLib/Scene/Transfer.h"

// Include Dependencies 
#include "mainwindow.h"
#include "transferitem.h"

// QT4 Dependencies
#include <QtGui/QGraphicsSceneMouseEvent>

// File scope namespace
namespace
{
    namespace filescope
    {
        // Stylesheet for button tooltip
        char const* toolTipTemplate = 
            "<table>                                                \
                <tr><td>Node</td><td>: </td><td>%1%</td></tr>       \
                <tr><td>Position</td><td>: </td><td>%2%</td></tr>   \
                <tr><td>Opacity</td><td>: </td><td>%3%</td></tr>    \
                <tr>                                                \
                    <td>Color</td><td>: </td>                       \
                    <td style='color:rgb(%4%,%5%,%6%)'><b>          \
                        <style type='text/css'>                     \
                            background {color:red;}                 \
                        </style>                                    \
                        [%4%,%5%,%6%]                               \
                    </b></td>                                       \
                </tr>                                               \
             </table>";

        // Epsilon value for node position adjustment
        const float nodePositionEpsilon = 0.01f;

        // Node item display parameters
        float	radius			= 4.0f;
        QBrush	brushNormal		= QBrush(QColor::fromHsl(0, 100, 150));
        QBrush	brushHighlight	= QBrush(QColor::fromHsl(0, 130, 150));
        QBrush	brushDisabled	= QBrush(QColor::fromHsl(0, 0, 230));

        QPen	penNormal		= QPen(QBrush(QColor::fromHsl(0, 100, 100)), 1.0);
        QPen	penHighlight	= QPen(QBrush(QColor::fromHsl(0, 150, 50)), 1.0);
        QPen	penDisabled		= QPen(QBrush(QColor::fromHsl(0, 0, 200)), 1.0);

        QBrush	selectionBrush	= QBrush(QColor::fromHsl(43, 150, 150, 255));
        QPen	selectionPen	= QPen(QBrush(QColor::fromHsl(0, 150, 100, 150)), 1.0);
    }
}

// --------------------------------------------------------------------
// Constructor - Initialize node as interactible object
// --------------------------------------------------------------------
NodeItem::NodeItem(TransferItem* parent, std::shared_ptr<vox::Node> node) : 
    QGraphicsEllipseItem(parent),
    m_parent(parent),
    m_pNode(node),
    m_ignorePosChange(false)
{
	setFlag( QGraphicsItem::ItemIsMovable );
	setFlag( QGraphicsItem::ItemSendsGeometryChanges );
	setFlag( QGraphicsItem::ItemIsSelectable );

    // Perform the initial scene rectangle update connection
    connect(parentItem()->scene(), SIGNAL(sceneRectChanged(QRectF)),
                this, SLOT(sceneRectangleChanged(QRectF)));

    // Update the tooltip
    setToolTip( vox::format( filescope::toolTipTemplate, 
        1, 2, 3, 4, 5, 6 ).c_str( ) );  // :TODO: :DEBUG: implement tooltip
};

// --------------------------------------------------------------------
//  Draws the node item graphic
// --------------------------------------------------------------------
void NodeItem::paint( QPainter* pPainter, 
	const QStyleOptionGraphicsItem* pOption, 
    QWidget* pWidget )
{
	QBrush Brush; QPen Pen;

	if (isEnabled())
	{
		if (isUnderMouse() || isSelected())
		{
			Brush = filescope::brushHighlight;
			Pen   = filescope::penHighlight;
		}
		else
		{
			Brush = filescope::brushNormal;
			Pen   = filescope::penNormal;
		}

		if (isUnderMouse() || isSelected())
		{
			QRectF SelRect = rect( );
			SelRect.adjust( -2.6, -2.6, 2.6, 2.6 );

			pPainter->setBrush( filescope::selectionBrush );
			pPainter->setPen( filescope::selectionPen );
			pPainter->drawEllipse( SelRect );
		}
	}
	else
	{
		Brush = filescope::brushDisabled;
		Pen   = filescope::penDisabled;
	}

    auto rectangle = rect();
	pPainter->setBrush( Brush );
	pPainter->setPen( Pen );
	pPainter->drawEllipse( rect( ) );
}

// --------------------------------------------------------------------
//  Sets the position of the node item
// --------------------------------------------------------------------
void NodeItem::setPos(const QPointF& newPos)
{
    if (newPos == this->pos()) return;

    QGraphicsEllipseItem::setPos(newPos);

    QRectF ellipseRect;

    ellipseRect.setTopLeft( QPointF(-filescope::radius, -filescope::radius) );
    ellipseRect.setWidth( 2.0f * filescope::radius );
    ellipseRect.setHeight( 2.0f * filescope::radius );

    setRect( ellipseRect );
}

// --------------------------------------------------------------------
//  Maintain link between node display item and actual node
// --------------------------------------------------------------------
QVariant NodeItem::itemChange(GraphicsItemChange change, const QVariant& value)
{
    QPointF position = value.toPointF();

    // Detect selection change due to 
    if (change == QGraphicsItem::ItemSelectedChange && !m_ignorePosChange)
    {
        bool selected = this->isSelected();

        if (!selected) MainWindow::instance->setTransferNode(m_pNode);
    }

    // Register a rectangle resize callback with the parent scene to 
    // ensure proper node positioning within the view
    if (change == QGraphicsItem::ItemSceneChange)
    {
        connect(parentItem()->scene(), SIGNAL(sceneRectChanged(QRectF)),
                this, SLOT(sceneRectangleChanged(QRectF)));
    }

    // Ensure the node remains within the parent boundaries
    if (change == QGraphicsItem::ItemPositionChange)
    {
        QPointF const nodeRangeMin = m_parent->rect( ).topLeft( );
        QPointF const nodeRangeMax = m_parent->rect( ).bottomRight( );

        position.setX( qMin( nodeRangeMax.x( ), qMax( position.x( ), nodeRangeMin.x( ) ) ) );
        position.setY( qMin( nodeRangeMax.y( ), qMax( position.y( ), nodeRangeMin.y( ) ) ) );

        return position;
    }

    // Update the associated transfer function node when moved
    if (change == QGraphicsItem::ItemPositionHasChanged && !m_ignorePosChange)
    {
        // :TODO: LOCKING MECHANISM (to prevent changes while mapping transfer) //
        
        float xNorm = (position.x() - m_parent->rect().x()) / m_parent->rect().width();
        float yNorm = (position.y() - m_parent->rect().y()) / m_parent->rect().height();
        
        // :TODO: Adjust parameter corresponding to parent transfer item's view 
        if (true) // Density + Opacity
        {
            m_pNode->setPosition(0, xNorm);
            m_pNode->material()->setOpticalThickness(yNorm);
        }
        else if (false) // Density + Gradient_Magnitude
        {
            m_pNode->setPosition(0, xNorm);
            m_pNode->setPosition(1, yNorm);
        }
        else // Density + Laplacian
        {
            m_pNode->setPosition(0, xNorm);
            m_pNode->setPosition(2, yNorm);
        }

        return position;
    }

    return QGraphicsItem::itemChange( change, value );
}

// --------------------------------------------------------------------
// Set the mouse cursor type when the node item is clicked
// --------------------------------------------------------------------
void NodeItem::mousePressEvent(QGraphicsSceneMouseEvent* pEvent)
{
	QGraphicsItem::mousePressEvent( pEvent );

	if( pEvent->button( ) == Qt::LeftButton )
		if( false ) setCursor( QCursor(Qt::SizeVerCursor) );
		else setCursor( QCursor(Qt::SizeAllCursor) );
}

// --------------------------------------------------------------------
// Return the cursor to normal when the node item is released
// --------------------------------------------------------------------
void NodeItem::mouseReleaseEvent(QGraphicsSceneMouseEvent* pEvent)
{
	QGraphicsEllipseItem::mouseReleaseEvent( pEvent );

	setCursor( QCursor(Qt::ArrowCursor) );
}

// --------------------------------------------------------------------
//  Updates the node item position when the parent scene
// --------------------------------------------------------------------
void NodeItem::sceneRectangleChanged(QRectF rectangle)
{
    QPointF normPos;

    // Determine the normalized view position based on transfer type
    if (true)  // :TODO: Analyze parent transfer's typee
    {
        normPos.setX(m_pNode->position(0));
        normPos.setY(m_pNode->material()->opticalThickness());
    }
    else if (false)
    {
    }
    else
    {
    }

    // Compute the realized coordinates of the node within its parent
    QPointF const nodeRangeMin = m_parent->rect( ).topLeft( );
    QPointF const nodeRangeMax = m_parent->rect( ).bottomRight( );
    QPointF newPos(
        nodeRangeMin.x() + (nodeRangeMax.x() - nodeRangeMin.x()) * normPos.x(),
        nodeRangeMin.y() + (nodeRangeMax.y() - nodeRangeMin.y()) * normPos.y()
        );

    // Issue the position update
    m_ignorePosChange = true;
    setPos(newPos); 
    m_ignorePosChange = false;
}