/* ===========================================================================

	Project: VoxRender - RenderView

    Copyright (C) 2012 Lucas Sherman

    Description: GraphicsView for visualizing render output

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
#ifndef RENDERVIEW_H
#define RENDERVIEW_H

// Qt4 Dependencies
#include <QtGui/QGraphicsView>
#include <QtGui/QGraphicsScene>
#include <QtGui/QApplication>
#include <QtCore/QEvent>
#include <QtGui/QGraphicsPixmapItem>
#include <QtGui/QWheelEvent>
#include <QtGui/QMouseEvent>
#include <QtGui/QMatrix>
#include <QtCore/QPoint>
#include <QtGui/QClipboard>

// VoxRender Dependencies
#include "VoxLib/Core/Geometry/Color.h"
#include "VoxLib/Rendering/FrameBuffer.h"

/** View portal for in progress render */
class RenderView : public QGraphicsView
{
	Q_OBJECT

public:
	~RenderView();

    /** Constructs a new render view portal */
	RenderView(QWidget *parent = 0);

    /** Enables image zoom operations with the portal */
	void setZoomEnabled(bool enabled = true)     { m_zoomEnabled = enabled; };

    /** Sets the flag for overlaying statistical data on the display image */
	void setOverlayStatistics(bool value = true) { m_overlayStats = value; };

    /** Returns the current image zoom factor */
	float zoomFactor() const { return m_zoomfactor; }

    /** Returns the current approximate framerate */
    float framerate() const { return m_framerate; }

    /** Copies the current image to the clipboard */
	void copyToClipboard() const; 

    /** Executes scene interactions */
    void processSceneInteractions();

public slots:
    /** Resets the display image */
    void sceneChanged( );

    /** Updates the image in the render view */
    void setImage(std::shared_ptr<vox::FrameBufferLock> lock);

private:
	void setLogoMode();
    void setViewMode();

private:
    bool  m_zoomEnabled;  ///< Flag indicating whether image zoom is enabled
    float m_zoomfactor;   ///< The current zoomfactor for the image zoom
    bool  m_overlayStats; ///< Flag indicating whether statistical overlay is enabled

	QGraphicsScene*      m_renderscene;   ///< Scene graph
	QGraphicsPixmapItem* m_voxlogo;       ///< Background logo for non-render mode display
	QGraphicsPixmapItem* m_voxfb;         ///< Active display image for the current render

    void wheelEvent(QWheelEvent* event);      ///< Handles image zoom operations
    void mousePressEvent(QMouseEvent* event); ///< Initiates tracking of mouse movement
    void mouseMoveEvent(QMouseEvent* event);  ///< Initiates tracking of mouse movement
    void resizeEvent(QResizeEvent* event);    ///< Handles resizing of the viewport
    void keyPressEvent(QKeyEvent * event);    ///< Detects keyboard-scene interaction
    void keyReleaseEvent(QKeyEvent * event);  ///< Detects release of interaction keys
    void focusOutEvent(QFocusEvent * event);  ///< Releases any held keys

    unsigned int m_ioFlags;     ///< Bitset for io tracking
    QPoint       m_prevPos;     ///< Mouse position after last event
    QAtomicInt   m_mouseDeltaX; ///< Mouse x coordinate change
    QAtomicInt   m_mouseDeltaY; ///< Mouse y coordinate change

    float m_framerate;  ///< The current approximate framerate
    float m_lastFrame;  ///< The timestamp for the last frame

signals:
	void viewChanged();
};

#endif // RENDERVIEW_H