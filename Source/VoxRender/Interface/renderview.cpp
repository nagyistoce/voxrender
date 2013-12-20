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

// Include Header
#include "renderview.h"

// Include Dependencies
#include "mainwindow.h"

// VoxRender Dependencies
#include "VoxLib/Core/Logging.h"
#include "VoxLib/Scene/Camera.h"

using namespace vox;

namespace {
namespace filescope {

    // Bit flags for tracking key events in the view
    static const unsigned int KEY_LEFT  = 1 << 0;
    static const unsigned int KEY_RIGHT = 1 << 1;
    static const unsigned int KEY_UP    = 1 << 2;
    static const unsigned int KEY_DOWN  = 1 << 3;

} // namespace filescope
} // namespace anonymous

// ------------------------------------------------------------
//  Constructor - initialize graphics scene for render view
// ------------------------------------------------------------
RenderView::RenderView(QWidget *parent) : 
    QGraphicsView(parent),
    m_renderscene(new QGraphicsScene()),
    m_overlayStats(false),
    m_ioFlags(0u)
{
	m_renderscene->setBackgroundBrush(QColor(127,127,127));
	
    setScene(m_renderscene);

	m_voxlogo = m_renderscene->addPixmap(QPixmap(":/images/voxlogo_bg.png"));
	m_voxfb   = m_renderscene->addPixmap(QPixmap(":/images/voxlogo_bg.png"));

    connect(MainWindow::instance, SIGNAL(sceneChanged()), this, SLOT(sceneChanged()));

    setDragMode(QGraphicsView::ScrollHandDrag);

	setLogoMode();
}

// ------------------------------------------------------------
//  Destructor - Free the view images
// ------------------------------------------------------------
RenderView::~RenderView() 
{
	delete m_voxfb;
	delete m_voxlogo;
	delete m_renderscene;
}

// ------------------------------------------------------------
//  Copies rendered image to clipboard
// ------------------------------------------------------------
void RenderView::copyToClipboard() const
{
	// Verify render mode
    if (!m_voxfb->isVisible()) 
    {
        Logger::addEntry(Severity_Warning, Error_MissingData, "GUI",
            "Copy to clipboard failed (no display image present)",
            __FILE__, __LINE__);
    }

    // Attempt to copy the display image to the board
    if (QClipboard *clipboard = QApplication::clipboard()) 
    {
        QImage image = m_voxfb->pixmap().toImage();
        clipboard->setImage(image.convertToFormat(QImage::Format_RGB32));
    }
    else
    {
        Logger::addEntry(Severity_Error, Error_System, "GUI",
            "Copy to clipboard failed (unable to open clipboard)",
            __FILE__, __LINE__);
    }
}

// ------------------------------------------------------------
//  Detects render context changes and updates the view
// ------------------------------------------------------------
void RenderView::sceneChanged() 
{
    setViewMode();
}

// ------------------------------------------------------------
//  Removes the render feed from the view 
// ------------------------------------------------------------
void RenderView::setLogoMode() 
{
    resetTransform();

    if (m_voxfb->isVisible()) m_voxfb->hide();

    if (!m_voxlogo->isVisible()) m_voxlogo->show();

    centerOn(m_voxlogo);

	setInteractive(false);
    m_zoomEnabled = false;
}

// ------------------------------------------------------------
//  Attaches render feed to view
// ------------------------------------------------------------
void RenderView::setViewMode() 
{
    m_lastTime = boost::chrono::high_resolution_clock::now();

    resetTransform();

    if (!m_voxfb->isVisible()) m_voxfb->show();

    if (m_voxlogo->isVisible()) m_voxlogo->hide();
    
    centerOn(m_voxfb);

	setInteractive(true);
    m_zoomEnabled = true;
}

// ------------------------------------------------------------
//  Update zoom factor on window resize event
// ------------------------------------------------------------
void RenderView::resizeEvent(QResizeEvent *event) 
{
	QGraphicsView::resizeEvent(event);
	emit viewChanged();
}

// ------------------------------------------------------------
//  Image zoom in/out on mouse wheel event (NOT camera zoom)
// ------------------------------------------------------------
void RenderView::wheelEvent(QWheelEvent* event) 
{
	if (!m_zoomEnabled) return;

	const float zoomsteps[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12.5, 17, 25, 33, 45, 50, 67, 75, 100, 
		125, 150, 175, 200, 250, 300, 400, 500, 600, 700, 800, 1000, 1200, 1600 };
	
	size_t numsteps = sizeof(zoomsteps) / sizeof(*zoomsteps);

	size_t index = std::min<size_t>(std::upper_bound(zoomsteps, zoomsteps + numsteps, m_zoomfactor) - zoomsteps, numsteps-1);
	if (event->delta( ) < 0) 
	{
		// if zoomfactor is equal to zoomsteps[index-1] we need index-2
		while (index > 0 && zoomsteps[--index] == m_zoomfactor);		
	}
	m_zoomfactor = zoomsteps[index];

	resetTransform();

	scale(m_zoomfactor / 100.f, m_zoomfactor / 100.f);

	emit viewChanged();
}

// ------------------------------------------------------------
//  Mouse press event handler - begin tracking mouse movement
// ------------------------------------------------------------
void RenderView::mousePressEvent(QMouseEvent* event) 
{
    if (event->button() == Qt::RightButton)
    {
        m_prevPos = event->pos();
    }

	// Parent class event handler
	QGraphicsView::mousePressEvent(event);
}

// ------------------------------------------------------------
//  Mouse move event handler - track movement distance if down
// ------------------------------------------------------------
void RenderView::mouseMoveEvent(QMouseEvent* event)
{
    if (event->buttons() & Qt::RightButton)
    {
        m_mouseDeltaX.fetchAndAddRelaxed(event->pos().x() - m_prevPos.x());
        m_mouseDeltaY.fetchAndAddRelaxed(event->pos().y() - m_prevPos.y());

        m_prevPos = event->pos();
    }

	// Parent class event handler
	QGraphicsView::mouseMoveEvent(event);
}

// ------------------------------------------------------------
//  Tracks scene interaction through key event input
// ------------------------------------------------------------
void RenderView::keyPressEvent(QKeyEvent * event)
{
    switch (event->key())
    {
        // Camera strafe keys
        case Qt::Key_W: m_ioFlags |= filescope::KEY_UP;    break;
        case Qt::Key_A: m_ioFlags |= filescope::KEY_LEFT;  break;
        case Qt::Key_S: m_ioFlags |= filescope::KEY_DOWN;  break;
        case Qt::Key_D: m_ioFlags |= filescope::KEY_RIGHT; break;
    }
}

// ------------------------------------------------------------
//  Tracks scene interaction through key event input
// ------------------------------------------------------------
void RenderView::keyReleaseEvent(QKeyEvent * event)
{
    switch (event->key())
    {
        // Camera strafe keys
        case Qt::Key_W: m_ioFlags &= ~filescope::KEY_UP;    break;
        case Qt::Key_A: m_ioFlags &= ~filescope::KEY_LEFT;  break;
        case Qt::Key_S: m_ioFlags &= ~filescope::KEY_DOWN;  break;
        case Qt::Key_D: m_ioFlags &= ~filescope::KEY_RIGHT; break;
    }
}

// ------------------------------------------------------------
//  Releases held keys when the view loses focus
// ------------------------------------------------------------
void RenderView::focusOutEvent(QFocusEvent * event)
{
    m_ioFlags = 0;
}

// ------------------------------------------------------------
//  Processes and applies scene interactions for this frame
// ------------------------------------------------------------
void RenderView::processSceneInteractions()
{
    static const float    camSpeed(5.0f);
    static const Vector2f camWSense(-0.0025f, 0.0025f);

    Camera & camera   = *MainWindow::instance->scene().camera;

    // Handle camera strafe associated with key holds
    float duration = 1.0f; // :TODO: Boost <chrono> read elapsed since last, track framerate etc 
    float distance = camSpeed * duration;
    if (m_ioFlags & filescope::KEY_UP)    camera.move(distance);
    if (m_ioFlags & filescope::KEY_RIGHT) camera.moveRight(distance);
    if (m_ioFlags & filescope::KEY_DOWN)  camera.move(-distance);
    if (m_ioFlags & filescope::KEY_LEFT)  camera.moveLeft(distance);

    // Handle camera rotation from mouse movement
    int dx = m_mouseDeltaX.fetchAndStoreRelaxed(0);
    int dy = m_mouseDeltaY.fetchAndStoreRelaxed(0);
    if (dx || dy)
    {
        Vector2f rotation = Vector2f(dx, dy) * camWSense;
        camera.pitch(rotation[1]); camera.yaw(rotation[0]);
    }
}

// ------------------------------------------------------------
//  Sets the display image for the render view
// ------------------------------------------------------------
void RenderView::setImage(std::shared_ptr<vox::FrameBufferLock> lock)
{
    static vox::Image<ColorRgbaLdr> image; // :TODO: This was introduced during a bug fix, move to member

    vox::FrameBuffer & frame = *lock->framebuffer.get();

    if (image.width() != frame.width() || image.height() != frame.height())
    {
        image = frame;

        auto rect = m_renderscene->sceneRect();
        m_renderscene->setSceneRect(
            0.0f, 0.0f, (float)image.width(),  (float)image.height());
    }
    else memcpy(image.data(), frame.data(), frame.size());

    QImage qimage((unsigned char*)image.data(),
        image.width(), image.height(),
        image.stride(), QImage::Format_RGB32);

    // Compute the performance statistics for this frame
    auto curr    = boost::chrono::high_resolution_clock::now();
    auto elapsed = curr - m_lastTime; 
    m_lastTime = curr;

    auto fps = 1000000.0f / boost::chrono::duration_cast<boost::chrono::microseconds>(elapsed).count();

    // Draws the statistical information overlay
    {
        QPainter painter;
        painter.begin(&qimage);
        painter.setPen(Qt::white);
        painter.setCompositionMode(QPainter::CompositionMode_Source);
 
        painter.drawText(2, 10, format("%1%",fps).c_str());  // Draw a number on the image
 
        painter.end();
    }

    m_voxfb->pixmap().convertFromImage(qimage);

    m_voxfb->update();
}