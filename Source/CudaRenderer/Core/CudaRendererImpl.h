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
#ifndef CR_CUDA_RENDERER_IMPL_H
#define CR_CUDA_RENDERER_IMPL_H

// Include Interface Definition
#include "CudaRenderer.h"

// Device representations of scene components
#include "CudaRenderer/Core/CBuffer.h"
#include "CudaRenderer/Core/CRandomBuffer.h"
#include "CudaRenderer/Core/CSampleBuffer.h"
#include "CudaRenderer/Scene/CLight.h"

// OptiX SDK Headers
#define NOMINMAX
#include <optixu/optixpp_namespace.h>

// API namespace
namespace vox 
{
    typedef CImgBuffer2D<ColorRgbaLdr> CLdrImgBuffer2D;
    typedef CBuffer1D<CLight>          CLightsBuffer1D;

/** Implements the CUDA device Renderer */
class CudaRendererImpl : public CudaRenderer
{
public:
    /** Initializes to the specified device */
    CudaRendererImpl(int device = 0);

    /** Virtual Destructor */
    virtual ~CudaRendererImpl();

    /** Set the active device */
    virtual void startup();

    /** Frees GPU Memory */
    virtual void shutdown();

    /** Reupload the scene data to the device */
    virtual void syncScene(Scene const& scene);

    /** Exports the scene data to the output resource */
    virtual void backupIpr(std::ostream & out) { }

    /** Merges the input image buffer with the internal one */
    virtual void pushIpr(IprImage const& ipr, size_t const& samples) { }

    /** Pulls the current in-progress-render buffer then clears it */
    virtual void pullIpr(IprImage & img, size_t & samples) { }

    /** Sets the render event callback function */
    virtual void setRenderEventCallback(RenderCallback callback)
    {
        boost::mutex::scoped_lock lock(m_mutex);

        m_callback = callback;
    }

    /** Execute the sampling kernels */
    virtual void render();

private:
    int m_device; /// Device ID

    optix::Context m_context; ///< Optix rendering context

    CLdrImgBuffer2D m_ldrBuffer;    ///< LDR post-processed image
    CSampleBuffer2D m_hdrBuffer;    ///< HDR raw sample data buffer
    CRandomBuffer2D m_rndSeeds0;    ///< Seed buffer for CUDA RNG
    CRandomBuffer2D m_rndSeeds1;    ///< Seed buffer for CUDA RNG
    CLightsBuffer1D m_lightBuffer;  ///< Array of scene lights

    optix::Material m_translucentMaterial;  ///< Optix material for volume rendering

    size_t m_filmWidth;  ///< Width of the framebuffer in pixels
    size_t m_filmHeight; ///< Height of the framebuffer in pixels

    RenderCallback m_callback; ///< User defined render callback
    boost::mutex   m_mutex;    ///< Mutex for callback modification

    std::shared_ptr<FrameBuffer> m_frameBuffer;  ///< Host side framebuffer

    // :DEBUG:
    void rebuildGeometry(); ///< Rebuilds scene geometry structures
};

}

// End definition
#endif // CR_CUDA_RENDERER_IMPL_H