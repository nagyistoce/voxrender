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

// Begin definition
#ifndef ANIMATE_ITEM_H
#define ANIMATE_ITEM_H

// QT Includes
#include <QtWidgets/QGraphicsView>
#include <QtWidgets/QGraphicsRectItem>

// Labeled grid graphics item
class AnimateItem : public QGraphicsRectItem
{
public:
	AnimateItem(QGraphicsItem * parent = nullptr);

	virtual void paint(QPainter* painter, 
		const QStyleOptionGraphicsItem* options, 
		QWidget* widget);

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
};

// End definition
#endif // ANIMATE_ITEM_H