/* ===========================================================================

	Project: VoxRender - Rendering model

	Description: Abstract base class which defines rendering modules

    Copyright (C) 2012 Lucas Sherman

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
#ifndef VOX_RENDERER_H
#define VOX_RENDERER_H

// VoxLib Dependencies
#include "VoxLib/Core/CudaCommon.h"
#include "VoxLib/Core/Geometry/Color.h"
#include "VoxLib/Rendering/FrameBuffer.h"
#include "VoxLib/Scene/Scene.h"

// API namespace
namespace vox 
{

typedef Image<ColorLabxHdr> IprImage; // Internal image format

/** Abstract Renderer */
class VOX_EXPORT Renderer
{
public:
    virtual ~Renderer() { }

    /** Startup Stage : Precedes the initiation of rendering */
    virtual void startup() {}

    /** Shutdown Stage : Postcedes the termination of rendering */
    virtual void shutdown() {}
    
    /** 
     * Exception Handler : Notifies the renderer of an exception 
     *
     * If an exception is caught in the thread object responsible
     * for this renderer, this function is called to resolve the issue. If 
     * the returned value is true, the controller will try to recover by
     * reinitiating the thread. Otherwise, the controller will
     * discontinue use of this thread without calling any other functions.
     */
    virtual bool exception(std::exception_ptr & exception);

    /** Synchronization Stage : Scene change propogation */
    virtual void syncScene(Scene const& scene, bool force = false) = 0;

    /** Raycasting stage : Generate samples */
    virtual void render() = 0;
};

/** Control Renderer for interactive display work */
class VOX_EXPORT MasterRenderer : virtual public Renderer
{
public:
    /**
     * In Progress Render Merge
     *
     * Called by the render controller when an in-progress image buffer
     * is available in the queue to be merged with the control Renderer's
     * global framebuffer copy.
     */
    virtual void pushIpr(IprImage const& ipr) = 0;

    /**
     * Provides the renderer with an output stream to which it should
     * send the raw image pixels in the Labx format, row major ordering.
     */ 
    virtual void backupIpr(std::ostream & out) = 0;
};

/** Control Renderer for interactive display work */
class VOX_EXPORT SlaveRenderer : virtual public Renderer
{
public:
    /** 
     * IPR pull : Pulls the in progress render buffer 
     *
     * This function is called by the render controller at an interval controlled by the 
     * renderer to merge the samples produced by this specific renderer with the global 
     * pool. The parameter buffer acts as an input and an output. On input the buffer 
     * contains the current in progress render. On output the buffer should contain the 
     * computed merge of the renderer's internal buffer with this buffer. Similarly, on 
     * input the samples parameter contains the number of samples per pixel in the input 
     * buffer and on output should reflect the number of samples per pixel in the output 
     * buffer.
     *
     * Following the completion of this function, the renderer should discard the written 
     * samples from its internal buffer. 
     */
    virtual void pullIpr(IprImage & img) = 0;
};


}

// End definition
#endif // VOX_RENDERER_H