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

// Include Dependencies
#include "mainwindow.h"
#include "animatewidget.h"
#include "VoxScene/Animator.h"

// QT Dependencies
#include <QtWidgets/QGraphicsSceneMouseEvent>

namespace {
namespace filescope {

} // namespace filescope
} // namespace anonymous

// ------------------------------------------------------------
// Constructor - initialize the grid graphic options
// ------------------------------------------------------------
AnimateItem::AnimateItem(AnimateWidget * parent) :
	QGraphicsRectItem(),
	m_brushEnabled(QBrush(QColor::fromHsl(0, 0, 80))),
	m_brushDisabled(QBrush(QColor::fromHsl(0, 0, 210))),
	m_penEnabled(QPen(QColor::fromHsl(0, 0, 80), 0.1)),
	m_penDisabled(QPen(QColor::fromHsl(0, 0, 190))),
    m_isMouseDown(false),
	m_font("Arial", 10),
    m_offset(0),
    m_range(120),
    m_step(10),
    m_mousePos(-1),
    m_framePos(0),
    m_parent(parent)
{
	setAcceptHoverEvents(true);
    setFlags(QGraphicsItem::ItemIsSelectable);

    connect(&m_scrollTimer, SIGNAL(timeout()), this, SLOT(scrollWindow()));
}

// ------------------------------------------------------------
//
// ------------------------------------------------------------
void AnimateItem::setFrame(int frame)
{
    m_framePos = frame;

    if (m_offset > frame) m_offset = frame;
    else if (m_offset + m_range < frame)  m_offset = frame - m_range;

    update();
}
 
// ------------------------------------------------------------
//  Scrolls the display window and then resets the timer
// ------------------------------------------------------------
void AnimateItem::scrollWindow()
{
    if (m_mousePos == 0) m_offset -= m_step;
    else m_offset += m_step;

    m_scrollTimer.start(200);

    update();
}

// ------------------------------------------------------------
//  Scrolls the window on mouse wheel events
// ------------------------------------------------------------
void AnimateItem::onMouseWheel(QWheelEvent * pEvent)
{
    m_offset += pEvent->delta() / 60;

    update();
}

// ------------------------------------------------------------
//
// ------------------------------------------------------------
void AnimateItem::onMousePress(QMouseEvent* pEvent)
{
    if (pEvent->button() == Qt::RightButton)
    {
        auto frame = m_mousePos + m_offset;

        auto animator = MainWindow::instance->scene()->animator;
        auto keys = animator->keyframes();
        auto iter = keys.begin();
        while (iter != keys.end() && iter->first < frame)
            ++iter;

        if (iter == keys.end() || iter->first != frame)
        {
           m_mousePos = -1;
        }
        else
        {
            m_dragIndex = frame;
            m_dragFrame = iter->second;
            animator->removeKeyframe(iter->first, true);
            m_isDragging = true;
            m_isMouseDown = true;
        }
    }
    else m_isMouseDown = true;

    update();
}

// ------------------------------------------------------------
//  Set the current frame to the hovered frame
// ------------------------------------------------------------
void AnimateItem::onMouseRelease(QMouseEvent* pEvent)
{
    m_isMouseDown = false;

    m_scrollTimer.stop();

    if (pEvent->button() == Qt::RightButton) 
    {
        if (m_isDragging)
        {
            auto animator = MainWindow::instance->scene()->animator;
            animator->addKeyframe(m_dragFrame, m_offset + m_mousePos);
            m_dragFrame.reset();
        }
    }
    else if (pEvent->button() == Qt::LeftButton)
    {
        m_parent->setFrame(m_mousePos + m_offset);
    }

    m_isDragging = false;

    update();
}

// ------------------------------------------------------------
//  Update the position of the draggable frame cursor
// ------------------------------------------------------------
void AnimateItem::onMouseMove(QMouseEvent* pEvent)
{
    // :TODO: Stuff to move to filescope/private members
	const float WIDTH  = 60.0f;
	const float Height = 18.0f;
    QRectF gridRect(rect());
    gridRect.setLeft(gridRect.left() + WIDTH);
    gridRect.setRight(gridRect.right() - 10.0f);
	const float DX = gridRect.width() / (float)m_range;

    // Compute the position of the mouse cursor
    auto pos = pEvent->pos();
    pos.setX(pos.x() - WIDTH);
    auto mousePos = (int)(pos.x() / DX + 0.5f);
    mousePos = vox::clamp<int>(mousePos, 0, m_range);
    
    // Determine if the cursor has moved
    if (mousePos == m_mousePos) return;

    m_mousePos = mousePos;
    m_parent->setFrameHover(m_mousePos + m_offset);

    // Scrolling
    if (m_isMouseDown)
    {
        m_scrollTimer.stop();
        if (m_mousePos == 0 || m_mousePos == m_range) 
            m_scrollTimer.start(500);
    }

    update();
}

// ------------------------------------------------------------
// ------------------------------------------------------------
void AnimateItem::onMouseEnter(QEvent * pEvent) { }

// ------------------------------------------------------------
//  Resets the mouse hover index
// ------------------------------------------------------------
void AnimateItem::onMouseLeave(QEvent * pEvent)
{
    m_mousePos = -1;
    m_parent->setFrameHover(-1);

    m_scrollTimer.stop();

    update();
}

// ------------------------------------------------------------
// Draws a 2D grid with the specified axial labels
// ------------------------------------------------------------
void AnimateItem::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
    char const * Y_LABELS[] = {"Camera", "Volume", "Transfer", "Clipping", "Lights", "Settings"};
    QColor       Y_COLORS[] = { Qt::red, Qt::lightGray, Qt::blue, Qt::green, Qt::darkGreen, Qt::darkCyan };

	const float Width  = 60.0f;
	const float Height = 18.0f;

    unsigned int m_numY = 6;  ///< Number of channels of interp'able scene data

    QRectF gridRect(rect());
    gridRect.setLeft(gridRect.left() + Width);
    gridRect.setRight(gridRect.right() - 10.0f);
    
    // Delta values for x and y axis markings
	const float DY = gridRect.height() / (float)m_numY;
	const float DX = gridRect.width()  / (float)(m_range / m_step);

	// Disable antialising for grid-line rendering
	painter->setRenderHint(QPainter::Antialiasing, false);

	// Draw backdrop
	painter->setBrush(QBrush(QColor::fromHsl(255, 255, 255)));
	painter->drawRect(gridRect);

	// Initialize painter settings
	if (isEnabled())
	{
		painter->setBrush(m_brushEnabled);
		painter->setPen(m_penEnabled);
	}
	else
	{
		painter->setBrush(m_brushDisabled);
		painter->setPen(m_penDisabled);
	}
	painter->setFont(m_font);

    // Determine the visible range of keyframes in the animator
    auto frames = MainWindow::instance->scene()->animator->keyframes();
    auto biter = frames.begin();
    while (biter != frames.end() && m_offset > biter->first) ++biter;
    auto eiter = biter;
    while (eiter != frames.end() && eiter->first <= m_offset + m_range) eiter++;

	// Draw markings along Y-axis 
	for (unsigned int i = 0; i < m_numY+1; i++)
	{
        auto h = gridRect.top()+i*DY;

        // Draw the trace lines
		if (i > 0 && i < m_numY) painter->drawLine(QPointF(gridRect.left(), h), QPointF(gridRect.right(), h));

        // Draw the scene component text labels
		painter->drawLine(QPointF(gridRect.left()-2, h), QPointF(gridRect.left(), h));
        if (i != m_numY)
		painter->drawText(QRectF(gridRect.left()-Width-5, gridRect.top()-0.5f*Height+(i+0.5f)*DY, Width, Height), 
                          Qt::AlignVCenter | Qt::AlignRight, 
                          Y_LABELS[i]);

        // Draw the interpolation points/lines for the keyframes
        if (i != m_numY)
        for (auto iter = biter; iter != eiter; ++iter)
        {
            auto w = gridRect.left() + (iter->first - m_offset) * DX / m_step;
            painter->drawEllipse(QPointF(w, h+0.5*DY), 5, 5);
            auto brush = painter->brush();
            painter->setBrush(QBrush(Y_COLORS[i]));
            painter->drawEllipse(QPointF(w, h+0.5*DY), 4, 4);
	        painter->setBrush(brush);
        }
	}

	// Draw markings along X-axis 
    int offset = (m_offset < 0) ? abs(m_offset) % 10 : (10 - (m_offset % 10)) % 10;
    float xoffset = DX * (offset / 10.0f);
	for (int i = 0; i <= m_range / m_step; i++)
	{
        auto w = gridRect.left() + i*DX + xoffset;
        
        // Draw the trace lines
        if (!offset || i != m_range / m_step) 
        {
            painter->drawLine(QPointF(w, gridRect.bottom()), QPointF(w, gridRect.top()));

		    // Draw frame number label
		    painter->drawLine(QPointF(w, gridRect.bottom()), QPointF(w, gridRect.bottom()+2));
		    painter->drawText(QRectF(w-0.5f*Width, gridRect.bottom()+5, Width, Height), 
                              Qt::AlignHCenter | Qt::AlignTop, 
                              QString::number(m_offset + offset + i*m_step));
        }
	}

    // Draw the mouse frame hover line if the mouse is captured
    if (m_mousePos != -1)
    {
        auto pen = m_isMouseDown ?
            QPen(QColor::fromHsl(127, 127, 127), 3) :
            QPen(QColor::fromHsl(127, 127, 127, 96), 3);
        painter->setPen(pen);
        auto xpos = gridRect.left() + DX * m_mousePos / ((float)m_step);
        painter->drawLine(QPointF(xpos, gridRect.bottom()), QPointF(xpos, gridRect.top()));
    }

    // Draw the frame drag ghost effect when repositioning a frame
    if (m_isDragging)
    {
        auto xpos = gridRect.left() + DX * (m_framePos - m_offset) / ((float)m_step);
    }

    // Draw the active frame line if in the view
    if (m_framePos >= m_offset && m_framePos <= m_offset + m_range)
    {
        painter->setPen(QPen(QColor::fromHsl(0, 0, 0), 3));
        auto xpos = gridRect.left() + DX * (m_framePos - m_offset) / ((float)m_step);
        painter->drawLine(QPointF(xpos, gridRect.bottom()), QPointF(xpos, gridRect.top()));
    }
}