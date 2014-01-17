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

// Begin definition
#ifndef VOX_RENDERER_CONTROLLER_H
#define VOX_RENDERER_CONTROLLER_H

// Include Dependencies
#include "VoxLib/Core/Common.h"
#include "VoxLib/Core/Geometry.h"
#include "VoxLib/Rendering/FrameBuffer.h"
#include "VoxLib/Rendering/Renderer.h"
#include "VoxLib/Rendering/RenderThread.h"
#include "VoxLib/Scene/Scene.h"

// API namespace
namespace vox
{
	/** Controller class for managing the rendering of a scene */
	class VOX_EXPORT RenderController
	{
    public:
        /** Typdefs for shared pointer renderers */
        typedef std::shared_ptr<MasterRenderer> MasterHandle;
        typedef std::shared_ptr<SlaveRenderer>  SlaveHandle;

        /** Template for render failure callback */
        typedef std::function<void()> ErrorCallback;

	public:	
        ~RenderController() { stop(); }

        /** Initializes a new render control device */
		RenderController() : m_isPaused(false), m_threadsChanged(false) { } 

		/** Returns the number of renderers in this controller */
		size_t numRenderers() const;

        /** Registers a new renderer with the controller */
        void addRenderer(SlaveHandle renderer);

        /** Removes a renderer from the controller */
        void removeRenderer(SlaveHandle renderer);

        /** Sets the transfer function that generates the scene transfer map */
        void setTransferFunction(std::shared_ptr<Transfer> transfer);

		/** Returns the scene currently being renderered */
        Scene const& currentScene() const { return m_scene; }

		/**
		 * Initiates rendering operations
         *
         * @param renderer      The Scene struct containing handles to scene components
         * @param scene         The Scene struct containing handles to scene components
         * @param iterations    The iteration number at which to stop rendering
         * @param errorCallback The callback function for controller exceptions
		 */
        void render(
            MasterHandle  renderer,
            Scene const&  scene, 
            size_t        iterations,
            ErrorCallback onError = nullptr
            );

        /** Returns the current number of iterations */
        size_t iterations() const;

        /** 
         * Terminates the render
         *
         * Stopping the render involves terminating all of the internal rendering threads. 
         * In order to restart the render, the restart method must be called. 
         */
        void stop();

        /** Pauses rendering */
        void pause();

        /** Resumes rendering */
        void unpause();
        
        /** Pause state accessor */
        bool isPaused() const;

        /** Render state accessor */
        bool isActive() const;

	private:
        friend RenderThread;

        void entryPoint(); ///< Control thread entry point

        std::list<std::shared_ptr<RenderThread>> m_renderThreads; ///< Renderer management threads

        boost::posix_time::time_duration m_backupRate; ///< Rate of IPR backup 

        MasterHandle  m_masterRenderer;   ///< Master renderer module for render operations 
        ErrorCallback m_errorCallback;    ///< User callback for master renderer failure
        size_t        m_targetIterations; ///< Targeted number of iterations
        size_t        m_currIterations;   ///< Current number of iterations
        Scene         m_scene;            ///< Handles to scene components 

        // Render control subroutines
        void handleError(std::exception_ptr & error);
        void managementSubroutine();
        void controlSubroutine();
        void synchronizationSubroutine();
        void imageUpdateSubroutine();
        void renderingSubroutine();
        void terminateRenderThreads();

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

}

// End definition
#endif // VOX_RENDERER_CONTROLLER_H