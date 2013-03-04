/* ===========================================================================

	Project: VoxRender - CUDA based Renderer

	Description: Implements a CUDA based Renderer

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
#ifndef CR_CUDA_RENDERER_H
#define CR_CUDA_RENDERER_H

// Common Library Header
#include "CudaRenderer/Core/Common.h"

// VoxLib Dependencies
#include "VoxLib/Core/Devices.h"
#include "VoxLib/Rendering/FrameBuffer.h"
#include "VoxLib/Rendering/Renderer.h"
#include "VoxLib/Scene/Scene.h"

// API namespace
namespace vox 
{

/** Implements the CUDA device renderer for master or slave usage */
class CR_EXPORT CudaRenderer : public MasterRenderer, public SlaveRenderer
{
public:
    /** Format of the post-render_frame callback function */
    typedef std::function<void(std::shared_ptr<FrameBuffer> frame)> RenderCallback;

    /** Factor function for CUDA device renderers */
    static std::shared_ptr<CudaRenderer> create();

    /**
     * Sets the render complete event callback function
     *
     * This callback function will be executed after completion of a successful
     * render operation but before the render() routine has returned. This provides
     * an opportunity to modify scene data and display the previously produced 
     * frame before subsequent rendering operations are resumed.
     */
    virtual void setRenderEventCallback(RenderCallback callback) = 0;
};

}

// End definition
#endif // CR_CUDA_RENDERER_H