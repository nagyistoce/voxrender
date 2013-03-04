/* ===========================================================================

	Project: VoxRender - Render Thread

	Description: Manages and the rendering thread for a single renderer

    Copyright (C) 2013 Lucas Sherman

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
#ifndef VOX_RENDER_THREAD_H
#define VOX_RENDER_THREAD_H

// Include Dependencies
#include "VoxLib/Core/Common.h"
#include "VoxLib/Core/Geometry.h"
#include "VoxLib/Rendering/FrameBuffer.h"
#include "VoxLib/Rendering/Renderer.h"
#include "VoxLib/Scene/Scene.h"

// API namespace
namespace vox {

    class RenderController;

/** Wrapper class for managing render thread resources */
class VOX_EXPORT RenderThread : public boost::noncopyable
{
public:
    /** Constructs control objects for this render thread */
    RenderThread(RenderController & controller, 
                 std::shared_ptr<Renderer> renderer) :
      m_controller(controller),
      m_renderer(renderer),
      m_failed(false)
    {
    }

    /** Calls the render threads terminate function */
    ~RenderThread() { terminateAndWait(); }

    /** Returns the associated render handle */
    std::shared_ptr<Renderer> renderer() { return m_renderer; }

    /** Initiates the render thread */
    void startup();

    /** Thread entry point for rendering */
    void entryPoint();

    /** Returns true if the renderer is active */
    bool active() const { return m_thread; }

    /** Returns true if the renderer is dead */
    bool failed() const { return m_failed; }

    /** Terminates execution of this renderer */
    void terminate() 
    { 
        boost::mutex::scoped_lock lock(m_mutex);

        if (!m_thread) return;

        m_thread->interrupt(); 
    }

    /** Combination of terminate() and wait() */
    void terminateAndWait() { terminate(); wait(); }

    /** Waits for completion of thread termination */
    void wait()
    {
        boost::mutex::scoped_lock lock(m_mutex);

        if (!m_thread) return;

        m_thread->join();
        m_thread.reset();
    }

private:
    RenderController & m_controller;

    // Render thread subroutines
    void handlePauseRequests(); 
    void handleError(std::exception_ptr & error);

    std::shared_ptr<Renderer>      m_renderer; ///< Render handle
    std::shared_ptr<boost::thread> m_thread;   ///< Thread handle
    boost::mutex                   m_mutex;    ///< Synchronization mutex

    bool m_failed;    ///< Render failure flag
};

} // namespace vox

#endif // VOX_RENDER_THREAD