/* ===========================================================================

	Project: GPU based Volume Scatter Renderer

    Description: A configurable renderer functioning as a master or slave. It
                 performs no, single, or multiple scattering integration 
                 using available GPU resources.

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
#ifndef VSR_VOLUME_SCATTER_RENDERER_H
#define VSR_VOLUME_SCATTER_RENDERER_H

// Common Library Header
#include "VolumeScatterRenderer/Core/Common.h"

// VoxLib Dependencies
#include "VoxLib/Core/Devices.h"
#include "VoxScene/FrameBuffer.h"
#include "VoxScene/Renderer.h"
#include "VoxScene/Scene.h"

// API namespace
namespace vox 
{

/** Implements the CUDA device renderer for master or slave usage */
class VSR_EXPORT VolumeScatterRenderer : public MasterRenderer, public SlaveRenderer
{
public:
    /** Format of the post-render_frame callback function */
    typedef std::function<void(std::shared_ptr<FrameBuffer> frame)> RenderCallback;

    /** Factor function for CUDA device renderers */
    static std::shared_ptr<VolumeScatterRenderer> create();

    /**
     * Sets the render complete event callback function
     *
     * This callback function will be executed after completion of a successful
     * render operation but before the render() routine has returned. This provides
     * an opportunity to modify scene data and display the previously produced 
     * frame before subsequent rendering operations are resumed.
     */
    virtual void setRenderEventCallback(RenderCallback callback) = 0;

    /** Sets the exposure factor for the tonemapping */
    virtual void setExposure(float exposure) = 0;

    /** Returns the runtime (in ms) of the rendering kernel */
    virtual float renderTime() = 0;
    
    /** Returns the runtime (in ms) of the tonemapping kernel */
    virtual float tonemapTime() = 0;

    /** Returns the runtime (in ms) of the rand seed generation */
    virtual float rndSeedTime() = 0;
};

}

// End definition
#endif // VSR_VOLUME_SCATTER_RENDERER_H