/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the <ORGANIZATION> nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// Begin definition
#ifndef GRID_ITEM_H
#define GRID_ITEM_H

// QT4 Includes
#include <QtGui/QGraphicsView>
#include <QtGui/QGraphicsRectItem>

// Labeled grid graphics item
class GridItem : public QGraphicsRectItem
{
public:
	GridItem( QGraphicsItem* parent = NULL );

	virtual void paint( QPainter* painter, 
		const QStyleOptionGraphicsItem* options, 
		QWidget* widget );

private:
	QBrush  m_bkBrushEnabled;	// Enabled state background brush
	QBrush	m_bkBrushDisabled;	// Disabled state background brush
	QBrush  m_bkPenEnabled;		// Enabled state background pen
	QBrush	m_bkPenDisabled;	// Disabled state background pen
	QBrush	m_brushEnabled;		// Enabled state brush type
	QBrush	m_brushDisabled;	// Disabled state brush type
	QPen	m_penEnabled;		// Enabled state pen type
	QPen	m_penDisabled;		// Disabled state pen type
	QFont	m_font;				// Font used for grid labels

	int		m_numY;				// Number of Y-axis gridlines / Y-axis spacing
	int		m_numX;				// Number of X-axis gridlines / X-axis spacing
	bool	m_isNum;			// Whether to render by number of spacing of lines
	bool	m_back;				// Whether the grid item has a backdrop
};

// End definition
#endif // GRID_ITEM_H