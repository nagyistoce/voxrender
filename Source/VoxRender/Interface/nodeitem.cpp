/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the <ORGANIZATION> nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// Modified for use with VoxRender // 

// Include Header
#include "nodeitem.h"

// VoxRender Includes
#include "VoxLib/Core/Format.h"

// Include Dependencies 
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

// ---------------------------------------------------------
// Constructor - Initialize node as interactible object
// ---------------------------------------------------------
NodeItem::NodeItem( TransferItem* parent ) : 
    QGraphicsEllipseItem( parent ),
    m_parent( parent )
{
	setFlag( QGraphicsItem::ItemIsMovable );
	setFlag( QGraphicsItem::ItemSendsGeometryChanges );
	setFlag( QGraphicsItem::ItemIsSelectable );

    // Update the tooltip
    setToolTip( vox::format( filescope::toolTipTemplate, 
        1, 2, 3, 4, 5, 6 ).c_str( ) );
};

// ---------------------------------------------------------
//  Draws the node item graphic
// ---------------------------------------------------------
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

	pPainter->setBrush( Brush );
	pPainter->setPen( Pen );
	pPainter->drawEllipse( rect( ) );
}

// ---------------------------------------------------------
//  Sets the position of the node item
// ---------------------------------------------------------
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

// ---------------------------------------------------------
//  Maintain link between node display item and actual node
// ---------------------------------------------------------
QVariant NodeItem::itemChange( GraphicsItemChange change, const QVariant& value )
{
    QPointF position = value.toPointF();

    // Ensure the node remains within the parent boundaries
    if( change == QGraphicsItem::ItemPositionChange )
    {
        QPointF const nodeRangeMin = m_parent->rect( ).topLeft( );
        QPointF const nodeRangeMax = m_parent->rect( ).bottomRight( );

        position.setX( qMin( nodeRangeMax.x( ), qMax( position.x( ), nodeRangeMin.x( ) ) ) );
        position.setY( qMin( nodeRangeMax.y( ), qMax( position.y( ), nodeRangeMin.y( ) ) ) );

        return position;
    }

    if( change == QGraphicsItem::ItemPositionHasChanged )
    {
        // LOCKING MECHANISM //

        // 1D Transfer node
        //m_pNode->SetIntensity( position.x( ) / m_parent->rect( ).width( ) );
        //m_pNode->SetOpacity( 1.0f - (position.y( ) / m_parent->rect( ).height( )) );

        return position;
    }

    return QGraphicsItem::itemChange( change, value );
}

// ---------------------------------------------------------
// Set the mouse cursor type when the node item is clicked
// ---------------------------------------------------------
void NodeItem::mousePressEvent( QGraphicsSceneMouseEvent* pEvent )
{
	QGraphicsItem::mousePressEvent( pEvent );

	if( pEvent->button( ) == Qt::LeftButton )
		if( false ) setCursor( QCursor(Qt::SizeVerCursor) );
		else setCursor( QCursor(Qt::SizeAllCursor) );
}

// ---------------------------------------------------------
// Return the cursor to normal when the node item is released
// ---------------------------------------------------------
void NodeItem::mouseReleaseEvent( QGraphicsSceneMouseEvent* pEvent )
{
	QGraphicsEllipseItem::mouseReleaseEvent( pEvent );

	setCursor( QCursor(Qt::ArrowCursor) );
}