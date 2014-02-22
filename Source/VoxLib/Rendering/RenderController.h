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

// Begin definition
#ifndef VOX_RENDERER_CONTROLLER_H
#define VOX_RENDERER_CONTROLLER_H

// Include Dependencies
#include "VoxLib/Core/Common.h"
#include "VoxLib/Core/Geometry.h"
#include "VoxLib/Rendering/Renderer.h"

// API namespace
namespace vox
{
    class Renderer;
    class Transfer;

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
        /** Destructor */
        ~RenderController();

        /** Initializes a new render control device */
		RenderController();

		/** Returns the number of renderers in this controller */
		size_t numRenderers() const;

        /** Registers a new renderer with the controller */
        void addRenderer(SlaveHandle renderer);

        /** Removes a renderer from the controller */
        void removeRenderer(SlaveHandle renderer);

        /** Sets the transfer function that generates the scene transfer map */
        void setTransferFunction(std::shared_ptr<Transfer> transfer);

		/** Returns the scene currently being renderered */
        Scene const& scene() const;

		/**
		 * Initiates rendering operations on a scene
         *
         * @param renderer      The master render module
         * @param scene         The scene to be rendered
         * @param iterations    The iteration number at which to stop rendering
         * @param errorCallback The callback function for controller exceptions
		 */
        void render(
            MasterHandle  renderer,
            Scene const&  scene, 
            size_t        iterations,
            ErrorCallback onError = nullptr
            );

		/**
		 * Initiates rendering operations on an animation sequence
         *
         * @param renderer      The master render module
         * @param animator      The keyframe animation sequence
         * @param iterations    The number of iterations to run PER FRAME
         * @param errorCallback The callback function for controller exceptions
		 */
        void render(
            MasterHandle  renderer,
            std::shared_ptr<Animator> animator, 
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

        /** Returns the total active time spent on the current render (in seconds) */
        long long renderTime() const;

	private:
        class Impl;
        Impl * m_pImpl;
	};

}

// End definition
#endif // VOX_RENDERER_CONTROLLER_H