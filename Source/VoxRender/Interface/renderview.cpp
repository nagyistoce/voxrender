/* ===========================================================================

	Project: VoxRender - RenderView

    Copyright (C) 2012-2014 Lucas Sherman

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
#include "infowidget.h"

// VoxRender Dependencies
#include "VoxLib/Core/Logging.h"
#include "VoxLib/Bitmap/Bitmap.h"
#include "VoxScene/Camera.h"
#include "VoxScene/Scene.h"
#include "VoxScene/PrimGroup.h"

#include <QDateTime>

using namespace vox;

namespace {
namespace filescope {

    // Bit flags for tracking key events in the view
    static const unsigned int KEY_LEFT  = 1 << 0;
    static const unsigned int KEY_RIGHT = 1 << 1;
    static const unsigned int KEY_FRONT = 1 << 2;
    static const unsigned int KEY_BACK  = 1 << 3;
    static const unsigned int KEY_SPINL = 1 << 4;
    static const unsigned int KEY_SPINR = 1 << 5;
    static const unsigned int KEY_UP    = 1 << 6;
    static const unsigned int KEY_DOWN  = 1 << 7;

} // namespace filescope
} // namespace anonymous

// ----------------------------------------------------------------------------
//  Constructor - initialize graphics scene for render view
// ----------------------------------------------------------------------------
RenderView::RenderView(QWidget *parent) : 
    QGraphicsView(parent),
    m_renderscene(new QGraphicsScene()),
    m_overlayStats(false),
    m_activeTool(Tool_Drag),
    m_ioFlags(0u),
    m_zoomfactor(1.f),
    m_displayMode(Stereo_Left)
{
	m_renderscene->setBackgroundBrush(QColor(127,127,127));
	
    setScene(m_renderscene);

	m_voxlogo = m_renderscene->addPixmap(QPixmap(":/images/voxlogo_bg.png"));
	m_voxfb   = m_renderscene->addPixmap(QPixmap(":/images/voxlogo_bg.png"));

    setDragMode(QGraphicsView::ScrollHandDrag);

	setLogoMode();
}

// ----------------------------------------------------------------------------
//  Destructor - Free the view images
// ----------------------------------------------------------------------------
RenderView::~RenderView() 
{
	delete m_voxfb;
	delete m_voxlogo;
	delete m_renderscene;
}

// ----------------------------------------------------------------------------
//  Copies rendered image to clipboard
// ----------------------------------------------------------------------------
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

// ----------------------------------------------------------------------------
//  Exports the contents of the render view to a file
// ----------------------------------------------------------------------------
void RenderView::saveImageToFile(String const& identifier) const
{
    try
    {
        m_image.exprt(identifier);
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

// ----------------------------------------------------------------------------
//  Removes the render feed from the view 
// ----------------------------------------------------------------------------
void RenderView::setLogoMode() 
{
    resetTransform();

    if (m_voxfb->isVisible()) m_voxfb->hide();

    if (!m_voxlogo->isVisible()) m_voxlogo->show();

    centerOn(m_voxlogo);

	setInteractive(false);
    m_zoomEnabled = false;
} 

// ----------------------------------------------------------------------------
//  Attaches render feed to view
// ----------------------------------------------------------------------------
void RenderView::setViewMode() 
{
    m_lastTime = std::chrono::high_resolution_clock::now();

    resetTransform();

    m_voxfb->show();
    m_voxlogo->hide();
    
    centerOn(m_voxfb);

	setInteractive(true);
    m_zoomEnabled = true;
}

// ----------------------------------------------------------------------------
//  Image zoom in/out on mouse wheel event (NOT camera zoom)
// ----------------------------------------------------------------------------
void RenderView::wheelEvent(QWheelEvent* event) 
{
	if (!m_zoomEnabled) return;

	const double zoomsteps[] = { 0.1, .125, 0.17, 0.25, 0.33, 0.45, 0.50, 0.67, 0.75, 1, 
		1.25, 1.5, 1.75, 2, 2.5, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20 };
	
	size_t numsteps = sizeof(zoomsteps) / sizeof(*zoomsteps);

	size_t index = std::min<size_t>(std::upper_bound(zoomsteps, zoomsteps + numsteps, m_zoomfactor) - zoomsteps, numsteps-1);
	if (event->delta( ) < 0) 
	{
		// if zoomfactor is equal to zoomsteps[index-1] we need index-2
		while (index > 0 && zoomsteps[--index] == m_zoomfactor);		
	}
	m_zoomfactor = (float)zoomsteps[index];

	resetTransform();

	scale(m_zoomfactor, m_zoomfactor);

	emit viewChanged(m_zoomfactor);
}

// ----------------------------------------------------------------------------
//  Mouse press event handler - begin tracking mouse movement
// ----------------------------------------------------------------------------
void RenderView::mousePressEvent(QMouseEvent* event) 
{
    if (event->button() == Qt::RightButton) m_prevPos = event->pos();
    
    // Handle the events for the active tool
    if (event->button() == Qt::LeftButton)
    {
        switch (m_activeTool)
        {
        case Tool_ClipPlane:
            auto scene = MainWindow::instance->scene();
            if (!scene->clipGeometry) return; 

            // Convert the position to image coordinates
            auto camera = MainWindow::instance->scene()->camera;
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
                    auto normal = Vector3f::cross(cv1, cv2).normalize();

                    // Compute the direction of the normal for clipping (towards the 3rd click)
                    auto sv1 = p2 - p1;
                    auto sv2 = p - p1;
                    if (sv1.x() * sv2.y() - sv1.y() * sv2.x() < 0) normal *= -1;

                    // Add the new clipping plane to the scene
                    auto plane = vox::Plane::create(normal, Vector3f::dot(normal, camera->position()));
                    scene->clipGeometry->add(plane);
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

// ----------------------------------------------------------------------------
//  Mouse move event handler - track movement distance if down
// ----------------------------------------------------------------------------
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

// ----------------------------------------------------------------------------
//  Tracks scene interaction through key event input
// ----------------------------------------------------------------------------
void RenderView::keyPressEvent(QKeyEvent * event)
{
    switch (event->key())
    {
        // Camera strafe keys
        case Qt::Key_W: m_ioFlags |= filescope::KEY_FRONT; break;
        case Qt::Key_A: m_ioFlags |= filescope::KEY_LEFT;  break;
        case Qt::Key_S: m_ioFlags |= filescope::KEY_BACK;  break;
        case Qt::Key_D: m_ioFlags |= filescope::KEY_RIGHT; break;
        case Qt::Key_Q: m_ioFlags |= filescope::KEY_SPINL; break;
        case Qt::Key_E: m_ioFlags |= filescope::KEY_SPINR; break;
        case Qt::Key_R: m_ioFlags |= filescope::KEY_UP;    break;
        case Qt::Key_F: m_ioFlags |= filescope::KEY_DOWN;  break;

        // :TEST: Plane cut
        case Qt::Key_Control: if (m_activeTool == Tool_Drag) m_activeTool = Tool_ClipPlane; break;
    }
}

// ----------------------------------------------------------------------------
//  Tracks scene interaction through key event input
// ----------------------------------------------------------------------------
void RenderView::keyReleaseEvent(QKeyEvent * event)
{
    switch (event->key())
    {
        // Camera strafe keys
        case Qt::Key_W: m_ioFlags &= ~filescope::KEY_FRONT; break;
        case Qt::Key_A: m_ioFlags &= ~filescope::KEY_LEFT;  break;
        case Qt::Key_S: m_ioFlags &= ~filescope::KEY_BACK;  break;
        case Qt::Key_D: m_ioFlags &= ~filescope::KEY_RIGHT; break;
        case Qt::Key_Q: m_ioFlags &= ~filescope::KEY_SPINL; break;
        case Qt::Key_E: m_ioFlags &= ~filescope::KEY_SPINR; break;
        case Qt::Key_R: m_ioFlags &= ~filescope::KEY_UP;    break;
        case Qt::Key_F: m_ioFlags &= ~filescope::KEY_DOWN;  break;

        // Plane cut
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

// ----------------------------------------------------------------------------
//  Releases held keys when the view loses focus
// ----------------------------------------------------------------------------
void RenderView::focusOutEvent(QFocusEvent * event)
{
    m_ioFlags = 0;
}

// ----------------------------------------------------------------------------
//  Sets the display mode used when stereo rendering is enabled
// ----------------------------------------------------------------------------
void RenderView::setDisplayMode(Stereo mode)
{
    m_displayMode = mode;
}

// ----------------------------------------------------------------------------
//  Processes and applies scene interactions for this frame
// ----------------------------------------------------------------------------
void RenderView::processSceneInteractions()
{
    static const float CAM_SPEED     = 5.0f;
    static const float CAM_ROT_SPEED = 0.0025f;

    // Compute the time statistics for speed scaling
    auto curr    = std::chrono::high_resolution_clock::now();
    auto elapsed = curr - m_lastTime; 
    m_lastTime = curr;
    m_lastTimeDiff = std::chrono::duration_cast<std::chrono::microseconds>(elapsed);

    // Perform the camera translation/rotation
    auto scene = MainWindow::instance->scene();
    auto camera = scene->camera;
    if (!camera) return;

    // -------
    // Process keyboard mappings
    if (m_ioFlags) 
    {
        float duration    = (float)m_lastTimeDiff.count() / 10000.0f;
        float distance    = CAM_SPEED * duration;
        float rotDistance = CAM_ROT_SPEED * duration;

        if (m_ioFlags & filescope::KEY_FRONT) camera->move(distance);
        if (m_ioFlags & filescope::KEY_RIGHT) camera->moveRight(distance);
        if (m_ioFlags & filescope::KEY_BACK)  camera->move(-distance);
        if (m_ioFlags & filescope::KEY_LEFT)  camera->moveLeft(distance);
        if (m_ioFlags & filescope::KEY_UP)    camera->moveUp(distance);
        if (m_ioFlags & filescope::KEY_DOWN)  camera->moveUp(-distance);
        if (m_ioFlags & filescope::KEY_SPINL) camera->roll(-rotDistance);
        if (m_ioFlags & filescope::KEY_SPINR) camera->roll(rotDistance);

        camera->setDirty();
    }

    // -------
    // Process mouse interactions
    int dx = m_mouseDeltaX.fetchAndStoreRelaxed(0);
    int dy = m_mouseDeltaY.fetchAndStoreRelaxed(0);
    if (dx || dy)
    {
        camera->pitch(dy * CAM_ROT_SPEED); 
        camera->yaw(dx * -CAM_ROT_SPEED);
        camera->setDirty();
    }
}

// ----------------------------------------------------------------------------
//  Sets the display image for the render view
// ----------------------------------------------------------------------------
void RenderView::setImage(std::shared_ptr<vox::FrameBufferLock> lock)
{
    vox::FrameBuffer & frame = *lock->framebuffer.get();

    // Image properties
    auto height = m_image.height();
    auto width  = m_image.width();
    auto stride = m_image.stride();
    
    // Set the scene rectangle
    auto sceneWidth = (m_displayMode == Stereo_SideBySide) ? width * 2 : width;
    m_renderscene->setSceneRect(0.0f, 0.0f, (float)sceneWidth, (float)height);

    // Stuff here is messed up, need to thoroughly go through the docs to figure out
    // how to do this properly
    if (width  != frame.width()  || 
        height != frame.height() || 
        m_image.layers() != frame.layers())
    {
        m_image = frame.copy();
    }
    else for (unsigned int i = 0; i < frame.layers(); i++)
    {
        memcpy(m_image.data(i), frame.data(i), frame.size());
    }

    QImage qimage;
    QPainter painter;

    // Compose the display image depending on the stereo settings
    switch (m_displayMode)
    {
    case Stereo_Left:
        qimage = QImage((unsigned char*)m_image.data(0),
            width, height, stride, QImage::Format_RGB32);
        break;
    case Stereo_Right:
        qimage = QImage((unsigned char*)m_image.data(m_image.layers() - 1),
            width, height, stride, QImage::Format_RGB32);
        break;
    case Stereo_SideBySide:
        qimage = QImage((unsigned char*)m_image.data(0),
            width, height, stride, QImage::Format_RGB32);
        qimage = qimage.copy(0, 0, width*2, height);
        
        painter.begin(&qimage);
        painter.drawImage(width, 0, 
            QImage((unsigned char*)m_image.data(m_image.layers() - 1),
                width, height, stride, QImage::Format_RGB32));
        break;
    default:
        qimage = QImage((unsigned char*)m_image.data(0),
            m_image.width(), m_image.height(),
            m_image.stride(), QImage::Format_RGB32);
        break;
    }

    // Compute the performance statistics for this frame
    auto fps = 1000000.0f / m_lastTimeDiff.count();

    // Draws the statistical information overlay
    if (m_overlayStats)
    {
        painter.begin(&qimage);
        painter.setPen(Qt::white);
        painter.setCompositionMode(QPainter::CompositionMode_Source);
 
        // Render: FPS      DATE        SAMPLES
        painter.drawText(2, 10, format("%1% fps",fps).c_str());
        painter.drawText(m_image.width() / 2 - 40, 10, QDateTime::currentDateTime().toString("yyyy-MM-dd"));
        painter.drawText(m_image.width() - 80, 10, format("%1% samples", 
            MainWindow::instance->m_renderController.iterations()).c_str());
 
        painter.end();
    }

    m_voxfb->pixmap().convertFromImage(qimage);
    
    m_voxfb->update();
}