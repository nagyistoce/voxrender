/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the <ORGANIZATION> nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// Include Header
#include "GridItem.h"

// ------------------------------------------------------------
// Constructor - initialize the grid graphic options
// ------------------------------------------------------------
GridItem::GridItem( QGraphicsItem* parent ) :
	QGraphicsRectItem( parent ),
	m_brushEnabled( QBrush(QColor::fromHsl(0, 0, 80)) ),
	m_brushDisabled( QBrush(QColor::fromHsl(0, 0, 210)) ),
	m_penEnabled( QPen(QColor::fromHsl(0, 0, 80), 0.1) ),
	m_penDisabled( QPen(QColor::fromHsl(0, 0, 190)) ),
	m_font( "Arial", 6 ),
	m_isNum( true ),
	m_numY( 10 ),
	m_numX( 10 )
{
	setAcceptHoverEvents( true );
}
    
// ------------------------------------------------------------
// Draws a 2D grid with the specified axial labels
// ------------------------------------------------------------
void GridItem::paint( QPainter* painter, 
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