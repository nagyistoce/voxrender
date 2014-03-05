/* ===========================================================================

	Project: VoxRender

	Description: Implements an animation keyframe display/interface tool

    Copyright (C) 2014 Lucas Sherman

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

// Include Header
#include "animateitem.h"

// ------------------------------------------------------------
// Constructor - initialize the grid graphic options
// ------------------------------------------------------------
AnimateItem::AnimateItem(QGraphicsItem * parent) :
	QGraphicsRectItem( parent ),
	m_brushEnabled( QBrush(QColor::fromHsl(0, 0, 80)) ),
	m_brushDisabled( QBrush(QColor::fromHsl(0, 0, 210)) ),
	m_penEnabled( QPen(QColor::fromHsl(0, 0, 80), 0.1) ),
	m_penDisabled( QPen(QColor::fromHsl(0, 0, 190)) ),
	m_font( "Arial", 6 )
{
	setAcceptHoverEvents(true);
}
    
// ------------------------------------------------------------
// Draws a 2D grid with the specified axial labels
// ------------------------------------------------------------
void AnimateItem::paint( QPainter* painter, 
	const QStyleOptionGraphicsItem* option, 
	QWidget* widget )
{
	// Disable antialising for grid-line rendering
	painter->setRenderHint( QPainter::Antialiasing, false );

	// Draw backdrop
	painter->setBrush( QBrush(QColor::fromHsl(255, 255, 255)) );
	painter->drawRect( rect( ) );

	// Initialize painter settings
	if( isEnabled( ) )
	{
		painter->setBrush( m_brushEnabled );
		painter->setPen( m_penEnabled );
	}
	else
	{
		painter->setBrush( m_brushDisabled );
		painter->setPen( m_penDisabled );
	}
	painter->setFont( m_font );

	const float Width = 25.0f;
	const float Height = 18.0f;

    unsigned int m_numX = 10; 
    unsigned int m_numY = 10; ///< Number of channels of interp'able scene data

	// Draw markings along Y-axis 
	const float DY = rect( ).height( ) / (float)m_numY;
	for( int i = 0; i < m_numY+1; i++ )
	{
		// Draw grid line
		if( i > 0 && i < m_numY )
			painter->drawLine( QPointF(rect().left(), rect().top()+i*DY), 
							   QPointF(rect().right(), rect().top()+i*DY) );

		// Draw axis label
		painter->drawLine( QPointF(rect().left()-2,rect().top()+i*DY), 
						   QPointF(rect().left(),rect().top()+i*DY) );
		painter->drawText( QRectF( rect().left()-Width-5, 
								   rect().top()-0.5f*Height+i*DY, 
							       Width, Height ), 
			Qt::AlignVCenter | Qt::AlignRight, 
			QString::number((10 - i) * 10.0f) + "%" );
	}

	// Draw markings along X-axis 
	const float DX = rect( ).width( ) / (float)m_numX;
	for( int i = 0; i < m_numX+1; i++ )
	{
		// Draw grid line
		if( i > 0 && i < m_numX )
			painter->drawLine( QPointF(rect().left()+i*DX, rect().bottom()), 
							   QPointF(rect().left()+i*DX, rect().top()) );

		// Draw label
		painter->drawLine( QPointF(rect().left()+i*DX, rect().bottom()), 
						   QPointF(rect().left()+i*DX, rect().bottom()+2) );
		painter->drawText( QRectF(rect().left()-0.5f*Width+i*DX, 
								  rect().bottom()+5, Width, Height ), 
			Qt::AlignHCenter | Qt::AlignTop, 
			QString::number(i * 10.0f) + "%");
	}
}