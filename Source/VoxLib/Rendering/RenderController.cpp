/* ===========================================================================

	Project: VoxLib

	Description: Manages the rendering of a scene

    Copyright (C) 2013-2014 Lucas Sherman

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

// Include Dependencies
#include "VoxLib/Core/Functors.h"
#include "VoxLib/Core/Geometry.h"
#include "VoxLib/Core/Logging.h"
#include "VoxLib/Scene/Camera.h"
#include "VoxLib/Scene/Light.h"
#include "VoxLib/Scene/PrimGroup.h"
#include "VoxLib/Scene/RenderParams.h"
#include "VoxLib/Scene/Scene.h"
#include "VoxLib/Scene/Transfer.h"
#include "VoxLib/Scene/Volume.h"
#include "VoxLib/Rendering/FrameBuffer.h"
#include "VoxLib/Rendering/Renderer.h"
#include "VoxLib/Rendering/RenderThread.h"

// Timing library
#include <chrono>

// API namespace
namespace vox
{

// Implementation class for render controller
class RenderController::Impl
{
public:
    // ----------------------------------------------------------------------------
    //  Initiates rendering of the currently loaded scene
    // ----------------------------------------------------------------------------
    void render(
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

        // Store the start time for render duration
        m_startTime = std::chrono::system_clock::now();

        // Launch the render control thread
        m_controlThread = std::shared_ptr<boost::thread>( 
            new boost::thread(std::bind(&Impl::entryPoint, this)));
    }

    // ----------------------------------------------------------------------------
    //  Returns the number of renderers currently registered
    // ----------------------------------------------------------------------------
    size_t numRenderers() const
    {
        return m_renderThreads.size();
    }

    // ----------------------------------------------------------------------------
    //  Returns the current number of iterations during rendering
    // ----------------------------------------------------------------------------
    size_t iterations() const
    {
        return m_currIterations;
    }
    
    // ----------------------------------------------------------------------------
    //  Returns the current scene being used for rendering
    // ----------------------------------------------------------------------------
    Scene const& scene() { return m_scene; }

    // ----------------------------------------------------------------------------
    //  Adds an additional renderer to the renderer list
    // ----------------------------------------------------------------------------
    void addRenderer(SlaveHandle renderer)
    {
        boost::mutex::scoped_lock lock(m_threadsMutex);

        m_renderThreads.push_back( std::make_shared<RenderThread>(renderer) );

        m_threadsChanged = true;
    }

    // ----------------------------------------------------------------------------
    //  Removes a renderer from the renderer list
    // ----------------------------------------------------------------------------
    void removeRenderer(SlaveHandle renderer)
    {
        boost::mutex::scoped_lock lock(m_threadsMutex);
    }

    // ----------------------------------------------------------------------------
    //  Changes the transfer map generator at runtime
    // ----------------------------------------------------------------------------
    void setTransferFunction(std::shared_ptr<Transfer> transfer)
    {
        m_scene.transfer = transfer; // :TODO: LOCK FOR CHANGE 
    }

    // ----------------------------------------------------------------------------
    //  Pauses the render controller
    // ----------------------------------------------------------------------------
    void pause()
    {
        m_isPaused = true;
    }   

    // ----------------------------------------------------------------------------
    //  Unpauses the render controller
    // ----------------------------------------------------------------------------
    void unpause() 
    { 
        m_isPaused = false; 

        m_pauseCond.notify_all(); 
    }

    // ----------------------------------------------------------------------------
    //  Returns true if the render controller is paused
    // ----------------------------------------------------------------------------
    bool isPaused() const 
    { 
        return m_isPaused;
    }

    // ----------------------------------------------------------------------------
    //  Unpauses the render controller
    // ----------------------------------------------------------------------------
    bool isActive() const
    { 
        return m_controlThread;
    }

    // ----------------------------------------------------------------------------
    //  Discontinues this controller's rendering operations
    // ----------------------------------------------------------------------------
    void stop()
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
    //  Returns the time spent on the current render
    // ----------------------------------------------------------------------------
    long long renderTime() const
    {
        auto duration = std::chrono::system_clock::now() - m_startTime;
        return std::chrono::duration_cast<std::chrono::seconds>(duration).count();
    }

private:
    // ----------------------------------------------------------------------------
    //  Rendering routine for the master/control thread
    // ----------------------------------------------------------------------------
    void entryPoint()
    {
        std::exception_ptr error = nullptr;

        try
        {
            m_masterRenderer->startup();

            synchronizationSubroutine(true);

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
    void terminateRenderThreads()
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
    void managementSubroutine()
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
    void synchronizationSubroutine(bool force = false)
    {
        if (force || m_scene.isDirty())
        {
            // Clone the scene data to prevent sync issues
            m_scene.clone(m_sceneCopy);

            // Synchronize the scene with the master renderer
            m_masterRenderer->syncScene(m_scene, force);

            m_scene.lightSet->setClean();
            m_scene.camera->setClean();
            m_scene.transfer->setClean();
            m_scene.parameters->setClean();

            m_scene.volume->m_isDirty = false;
            m_scene.clipGeometry->setDirty(false);
            m_scene.transferMap->setDirty(false);

            // Reset the rendering timestamp
            m_startTime = std::chrono::system_clock::now();
            m_currIterations = 0;
        }
    }

    // ----------------------------------------------------------------------------
    //  Routine for image update / synchronization 
    // ----------------------------------------------------------------------------
    void imageUpdateSubroutine()
    {
    }

    // ----------------------------------------------------------------------------
    //  Routine for the render control operations
    // ----------------------------------------------------------------------------
    void controlSubroutine()
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
    void renderingSubroutine()
    {
        m_masterRenderer->render();
    }

    // ----------------------------------------------------------------------------
    //  Executes the user error handling as necessary 
    // ----------------------------------------------------------------------------
    void handleError(std::exception_ptr & error)
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

private:
    std::list<std::shared_ptr<RenderThread>> m_renderThreads; ///< Renderer management threads
 
    std::chrono::system_clock::time_point m_startTime;  ///< Start time of the current render

    MasterHandle  m_masterRenderer;   ///< Master renderer module for render operations 
    ErrorCallback m_errorCallback;    ///< User callback for master renderer failure
    size_t        m_targetIterations; ///< Targeted number of iterations
    size_t        m_currIterations;   ///< Current number of iterations
    Scene         m_scene;            ///< Handles to scene components 
    Scene         m_sceneCopy;        ///< Handle to scene for rendering operations

    // ControlThread syncronization context
    std::shared_ptr<boost::thread> m_controlThread;
    boost::mutex                   m_controlMutex;

    // Pause synchronization context
    bool            m_isPaused;
    boost::mutex    m_pauseMutex;
    boost::cond_var m_pauseCond;              

    // RenderThread list synchronization context
    mutable boost::mutex m_threadsMutex;  
    bool                 m_threadsChanged;
};

// Redirect functions to the implementation class
RenderController::~RenderController() { stop(); delete m_pImpl; }
RenderController::RenderController() : m_pImpl(new Impl) { } 
size_t RenderController::numRenderers() const { return m_pImpl->numRenderers(); }
void RenderController::addRenderer(SlaveHandle renderer) { m_pImpl->addRenderer(renderer); }
void RenderController::removeRenderer(SlaveHandle renderer) { m_pImpl->removeRenderer(renderer); }
void RenderController::setTransferFunction(std::shared_ptr<Transfer> transfer) { m_pImpl->setTransferFunction(transfer); }
Scene const& RenderController::scene() const { return m_pImpl->scene(); }
void RenderController::render(MasterHandle renderer, Scene const& scene, 
                              size_t iterations, ErrorCallback onError) 
{ 
    m_pImpl->render(renderer, scene, iterations, onError); 
}
size_t RenderController::iterations() const { return m_pImpl->iterations(); }
void RenderController::stop() { m_pImpl->stop(); }
void RenderController::pause() { m_pImpl->pause(); }
void RenderController::unpause() { m_pImpl->unpause(); }
bool RenderController::isPaused() const { return m_pImpl->isPaused(); }
bool RenderController::isActive() const { return m_pImpl->isActive(); }
long long RenderController::renderTime() const { return m_pImpl->renderTime(); }

} // namespace vox