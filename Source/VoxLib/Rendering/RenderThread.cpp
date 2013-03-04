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

// Include Header
#include "RenderThread.h"

// API Includes
#include "VoxLib/Core/Devices.h"
#include "VoxLib/Core/Functors.h"
#include "VoxLib/Core/Geometry.h"
#include "VoxLib/Core/Logging.h"
#include "VoxLib/Rendering/RenderController.h"
#include "VoxLib/Scene/Film.h"
#include "VoxLib/Scene/Volume.h"
#include "VoxLib/Scene/Camera.h"

// API namespace
namespace vox
{

// --------------------------------------------------------------------
//  Initiates rendering on the specified thread
// --------------------------------------------------------------------
void RenderThread::startup()
{
    boost::mutex::scoped_lock lock(m_mutex);

    if (m_thread)
    {
        throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY,
                    "RenderThread already active", Error_NotAllowed);
    }

    m_failed = false;

    m_thread = std::make_shared<boost::thread>(
        std::bind(&RenderThread::entryPoint, this));
}

// --------------------------------------------------------------------
//  Entry point for the rendering thread
// --------------------------------------------------------------------
void RenderThread::entryPoint()
{
    std::exception_ptr error = nullptr;

    try
    {
        m_renderer->startup(); // Notify render initiation

        m_renderer->syncScene(m_controller.m_scene); // Initialize scene context

        while (true)
        {
            boost::this_thread::interruption_point();

            // Update scene info
            //m_renderer->syncScene(m_controller.m_scene);

            m_renderer->render(); // Generate additional samples

            // if (m_renderer->render())
            // {
            //     if (!m_controller.m_queueFree.tryPop(buffer))
            //     {
            //         buffer = new RenderController::ImageIPR(dimensions);
            //     }
            //
            //     m_renderer->extractIPR(buffer); // Extract in progress render
            //     m_controller.m_queueIPR.push(buffer); // Pass render frame to controller
            // }

            // Respond to user pause requests
            handlePauseRequests();
        }
    }
    catch (boost::thread_interrupted &)
    {
        m_renderer->shutdown(); // Notify render termination
    }
    catch (...)
    {
        error = std::current_exception();
    }

    handleError(error);
}

// --------------------------------------------------------------------
//  Checks for pause requests in the render controller
// --------------------------------------------------------------------
void RenderThread::handlePauseRequests()
{
    if (m_controller.m_isPaused)
    {
        boost::mutex::scoped_lock lock(m_controller.m_pauseMutex);

        while (m_controller.m_isPaused) 
        {
            m_controller.m_pauseCond.wait(lock);
        }
    }
}

// --------------------------------------------------------------------
//  Perform user exception callback on failure
// --------------------------------------------------------------------
void RenderThread::handleError(std::exception_ptr & error)
{
    if (!(error == nullptr))
    {
        boost::mutex::scoped_lock lock(m_mutex);

        m_thread.reset(); // Mark render thread terminated

        try
        {
            // Check for restart flag and notify controller if set
            if (!(m_failed = m_renderer->exception(error)))
            {
                m_controller.m_threadsChanged = true;
            }
        }
        catch(...)
        {
            Logger::addEntry(Severity_Error, Error_NotAllowed, VOX_LOG_CATEGORY,
                             "Terminating Renderer: Exception thrown by exception handler",
                             __FILE__, __LINE__);

            m_failed = true;
        }

    }
}

} // namespace vox