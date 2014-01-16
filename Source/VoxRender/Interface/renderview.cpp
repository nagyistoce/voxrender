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
#include "VoxLib/Image/RawImage.h"
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
    m_activeTool(Tool_Drag),
    m_ioFlags(0u),
    m_zoomfactor(1.f)
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
//  Exports the contents of the render view to a file
// ------------------------------------------------------------
void RenderView::saveImageToFile(String const& identifier) const
{
    try
    {
        QImage image = m_voxfb->pixmap().toImage();
        auto type = image.format();
        RawImage(RawImage::Format_RGBX, image.width(), image.height(), 
            8, image.bytesPerLine(), std::shared_ptr<void>((void*)image.bits(), [](void* p){}))
            .exprt(identifier);
    }
    catch (Error & error)
    {
        VOX_LOG_EXCEPTION(Severity_Error, error);
    }
    catch(...)
    {
        VOX_LOG_ERROR(Error_Unknown, VOX_LOG_CATEGORY, "An unknown error occurred while saving image file.");
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

	const float zoomsteps[] = { 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, .125, 0.17, 0.25, 0.33, 0.45, 0.50, 0.67, 0.75, 1, 
		1.25, 1.5, 1.75, 2, 2.5, 3, 4, 5, 6, 7, 8, 10, 12, 16 };
	
	size_t numsteps = sizeof(zoomsteps) / sizeof(*zoomsteps);

	size_t index = std::min<size_t>(std::upper_bound(zoomsteps, zoomsteps + numsteps, m_zoomfactor) - zoomsteps, numsteps-1);
	if (event->delta( ) < 0) 
	{
		// if zoomfactor is equal to zoomsteps[index-1] we need index-2
		while (index > 0 && zoomsteps[--index] == m_zoomfactor);		
	}
	m_zoomfactor = zoomsteps[index];

	resetTransform();

	scale(m_zoomfactor, m_zoomfactor);

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
    
    // Handle the events for the active tool
    if (event->button() == Qt::LeftButton)
    {
        switch (m_activeTool)
        {
        case Tool_ClipPlane:
            // Convert the position to image coordinates
            auto camera = MainWindow::instance->scene().camera;
            auto p = event->pos() - viewport()->mapToParent(mapFromScene(0, 0));
            p /= m_zoomfactor;

            // Store the click position until we have enough data to generate the clip plane
            if      (m_clipLine.p1().isNull()) m_clipLine.setP1(p); 
            else if (m_clipLine.p2().isNull()) m_clipLine.setP2(p);
            else 
            {
                // Construct the clipping plane
                if (m_clipLine.p1() != m_clipLine.p2()) 
                {
                    auto p1 = m_clipLine.p1();
                    auto p2 = m_clipLine.p2();

                    // Compute the plane normal vector
                    auto cv1 = camera->projectRay(Vector2f(p1.x(), p1.y())).dir;
                    auto cv2 = camera->projectRay(Vector2f(p2.x(), p2.y())).dir;
                    auto normal = Vector3f::cross(cv1, cv2);

                    // Compute the direction of the normal for clipping (towards the 3rd click)
                    auto sv1 = p2 - p1;
                    auto sv2 = p - p1;
                    if (sv1.x() * sv2.y() - sv1.y() * sv2.x() < 0) normal *= -1;

                    // Add the new clipping plane to the scene
                    auto plane = vox::Plane::create(normal, Vector3f::dot(normal, camera->position()));
                    MainWindow::instance->addClippingGeometry(plane);
                }
                m_clipLine.setP1(QPoint());
                m_clipLine.setP2(QPoint());
            }
            break;
        }
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

        // :TEST: Plane cut
        case Qt::Key_Control: if (m_activeTool == Tool_Drag) m_activeTool = Tool_ClipPlane; break;
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

        // :TEST: Plane cut
        case Qt::Key_Control: 
            if (m_activeTool == Tool_ClipPlane) 
            {
                m_activeTool = Tool_Drag; 
                m_clipLine.setP1(QPoint());
                m_clipLine.setP2(QPoint());
            }
            break;
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
    float duration = 1.0f;
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
    static vox::Image<vox::ColorRgbaLdr> m_image;

    MainWindow::instance->infowidget->updatePerformanceStatistics();

    vox::FrameBuffer & frame = *lock->framebuffer.get();

    if (m_image.width() != frame.width() || m_image.height() != frame.height())
    {
        m_image = frame.copy();

        auto rect = m_renderscene->sceneRect();
        m_renderscene->setSceneRect(
            0.0f, 0.0f, (float)m_image.width(),  (float)m_image.height());
    }
    else memcpy(m_image.data(), frame.data(), frame.size());

    QImage qimage((unsigned char*)m_image.data(),
        m_image.width(), m_image.height(),
        m_image.stride(), QImage::Format_RGB32);

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