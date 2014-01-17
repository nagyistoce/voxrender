/* ===========================================================================

	Project: VoxRender - Render Controller

	Description: Manages and controls a set of renderers for a common scene

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
#include "RenderController.h"

// API Includes
#include "VoxLib/Core/Devices.h"
#include "VoxLib/Core/Functors.h"
#include "VoxLib/Core/Geometry.h"
#include "VoxLib/Core/Logging.h"
#include "VoxLib/Scene/Camera.h"
#include "VoxLib/Scene/Light.h"
#include "VoxLib/Scene/Transfer.h"
#include "VoxLib/Scene/Volume.h"
#include "VoxLib/Scene/RenderParams.h"
#include "VoxLib/Scene/PrimGroup.h"

// API namespace
namespace vox
{

// ----------------------------------------------------------------------------
//  Initiates rendering of the currently loaded scene
// ----------------------------------------------------------------------------
void RenderController::render(
    MasterHandle  renderer,
    Scene const&  scene, 
    size_t        iterations,
    ErrorCallback onError
    )
{
    boost::mutex::scoped_lock lock(m_controlMutex);

    // Check for an in progress rendering operation
    if (m_controlThread)
    {
        throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY,
            "Render operation already in progress", Error_NotAllowed);
    }    
    
    // Check for missing renderer handle
    if (!renderer)
    {
        throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY,
            "Missing master renderer handle", Error_NotAllowed);
    }

    scene.issueWarningsForMissingHandles(); // Context warnings

    // Initialize the execution configuration
    m_masterRenderer   = renderer;
    m_targetIterations = iterations;
    m_errorCallback    = onError;
    m_scene            = scene;
    m_isPaused         = false;
    m_currIterations   = 0;

    // Launch the render control thread
    m_controlThread = std::shared_ptr<boost::thread>( 
        new boost::thread(std::bind(&RenderController::entryPoint, this)));
}

// ----------------------------------------------------------------------------
//  Returns the number of renderers currently registered
// ----------------------------------------------------------------------------
size_t RenderController::numRenderers() const
{
    return m_renderThreads.size();
}

// ----------------------------------------------------------------------------
//  Returns the current number of iterations during rendering
// ----------------------------------------------------------------------------
size_t RenderController::iterations() const
{
    return m_currIterations;
}

// ----------------------------------------------------------------------------
//  Adds an additional renderer to the renderer list
// ----------------------------------------------------------------------------
void RenderController::addRenderer(SlaveHandle renderer)
{
    boost::mutex::scoped_lock lock(m_threadsMutex);

    m_renderThreads.push_back( std::make_shared<RenderThread>(*this, renderer) );

    m_threadsChanged = true;
}

// ----------------------------------------------------------------------------
//  Removes a renderer from the renderer list
// ----------------------------------------------------------------------------
void RenderController::removeRenderer(SlaveHandle renderer)
{
    boost::mutex::scoped_lock lock(m_threadsMutex);
}

// ----------------------------------------------------------------------------
//  Changes the transfer map generator at runtime
// ----------------------------------------------------------------------------
void RenderController::setTransferFunction(std::shared_ptr<Transfer> transfer)
{
    m_scene.transfer = transfer; // :TODO: LOCK FOR CHANGE 
}

// ----------------------------------------------------------------------------
//  Pauses the render controller
// ----------------------------------------------------------------------------
void RenderController::pause()
{
    m_isPaused = true;
}   

// ----------------------------------------------------------------------------
//  Unpauses the render controller
// ----------------------------------------------------------------------------
void RenderController::unpause() 
{ 
    m_isPaused = false; 

    m_pauseCond.notify_all(); 
}

// ----------------------------------------------------------------------------
//  Returns true if the render controller is paused
// ----------------------------------------------------------------------------
bool RenderController::isPaused() const 
{ 
    return m_isPaused;
}

// ----------------------------------------------------------------------------
//  Unpauses the render controller
// ----------------------------------------------------------------------------
bool RenderController::isActive() const
{ 
    return m_controlThread;
}

// ----------------------------------------------------------------------------
//  Discontinues this controller's rendering operations
// ----------------------------------------------------------------------------
void RenderController::stop()
{
    boost::mutex::scoped_lock lock(m_controlMutex);

    if (m_controlThread)
    {
        m_controlThread->interrupt();
        m_controlThread->join(); 
        m_controlThread.reset();
        m_scene.reset();
    }
}

// ----------------------------------------------------------------------------
//  Rendering routine for the master/control thread
// ----------------------------------------------------------------------------
void RenderController::entryPoint()
{
    std::exception_ptr error = nullptr;

    try
    {
        m_masterRenderer->startup();

        synchronizationSubroutine();

        while (m_targetIterations > m_currIterations)
        {
            boost::this_thread::interruption_point();

            managementSubroutine();      // Manages render threads

            synchronizationSubroutine(); // Synchronize scene changes

            imageUpdateSubroutine();     // Pull and push IPRs

            renderingSubroutine();       // Perform master rendering
            
            controlSubroutine();         // General control checks

            m_currIterations++;
        }
    }
    catch (boost::thread_interrupted &)
    {
        // Do not treat interrupts as errors //
    }
    catch (...)
    {
        error = std::current_exception();
    }

    handleError(error); // User exception handling

    // Render thread cleanup
    terminateRenderThreads();
}

// ----------------------------------------------------------------------------
//  Terminate render threads 
// ----------------------------------------------------------------------------
void RenderController::terminateRenderThreads()
{
    m_masterRenderer->shutdown(); //:TODO: Handle exception

    BOOST_FOREACH (auto & thread, m_renderThreads)
    {
        thread->terminate();
    }
    BOOST_FOREACH (auto & thread, m_renderThreads)
    {
        thread->wait();
    }

    m_masterRenderer.reset();
}

// ----------------------------------------------------------------------------
//  Routine for renderer thread management - Revives and 
//  dead render threads and performs startup for new ones
// ----------------------------------------------------------------------------
void RenderController::managementSubroutine()
{
    if (m_threadsChanged)
    {
        boost::mutex::scoped_lock lock(m_threadsMutex);

        BOOST_FOREACH (auto & rthread, m_renderThreads)
        {
            if (!rthread->active() && !rthread->failed())
            {
                rthread->startup();
            }
        }

        m_threadsChanged = false; 
    }
}

// ----------------------------------------------------------------------------
//  Routine for renderer thread management
// ----------------------------------------------------------------------------
void RenderController::synchronizationSubroutine()
{
    if (m_scene.camera->isDirty() ||
        m_scene.lightSet->isDirty() ||
        m_scene.parameters->isDirty() ||
        m_scene.transfer->isDirty() ||
        m_scene.clipGeometry->isDirty())
    {
        m_masterRenderer->syncScene(m_scene);

        m_scene.camera->m_contextChanged     = false;
        m_scene.camera->m_filmChanged        = false;
        m_scene.lightSet->m_contextChanged   = false;
        m_scene.lightSet->m_contentChanged   = false;
        m_scene.lightSet->m_ambientChanged   = false;
        m_scene.volume->m_contextChanged     = false;
        m_scene.transfer->m_contextChanged   = false;
        m_scene.transfer->m_contextChanged   = false;
        m_scene.parameters->m_contextChanged = false;
        m_scene.clipGeometry->setDirty(false);

        m_currIterations = 0;
    }
}

// ----------------------------------------------------------------------------
//  Routine for image update / synchronization 
// ----------------------------------------------------------------------------
void RenderController::imageUpdateSubroutine()
{
}

// ----------------------------------------------------------------------------
//  Routine for the render control operations
// ----------------------------------------------------------------------------
void RenderController::controlSubroutine()
{
    // Respond to user pause requests
    if (m_isPaused)
    {
        boost::mutex::scoped_lock lock(m_pauseMutex);
        while (m_isPaused) m_pauseCond.wait(lock);
    }
}

// ----------------------------------------------------------------------------
//  Performs the user frame display/processing callback
// ----------------------------------------------------------------------------
void RenderController::renderingSubroutine()
{
    m_masterRenderer->render();
}

// ----------------------------------------------------------------------------
//  Executes the user error handling as necessary 
// ----------------------------------------------------------------------------
void RenderController::handleError(std::exception_ptr & error)
{
    if (!(error == nullptr))
    {
        try
        {
            m_masterRenderer->exception(error);
        }
        catch(...)
        {
            Logger::addEntry(Severity_Fatal, Error_NotAllowed, VOX_LOG_CATEGORY,
                             "Exception thrown by master exception handler", 
                             __FILE__, __LINE__);
        }
    }
}

} // namespace vox