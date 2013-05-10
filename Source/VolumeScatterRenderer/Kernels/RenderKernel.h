/* ===========================================================================

	Project: GPU based Volume Scatter Renderer
    
	Description: Implements CUDA based volume rendering

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
#ifndef VSR_RENDER_KERNEL_H
#define VSR_RENDER_KERNEL_H

// Common Library Header
#include "VolumeScatterRenderer/Core/Common.h"

// API namespace
namespace vox 
{
    
// Forward Declarations
template<typename T> class CBuffer1D;
template<typename T> class CImgBuffer2D;

class CRenderParams;
class CSampleBuffer2D;
class CRandomBuffer2D;
class CTransferBuffer;
class CVolumeBuffer;
struct ColorRgbaLdr;
class CLight;
class CCamera;
class LightSet;

/** Defines the interface for building and executing render kernels */
class RenderKernel
{
public:
    /** Sets the camera model for the render kernel */
    static void setCamera(CCamera const& camera);
    
    /** Sets the scene lightbuffer to the specified buffer */
    static void setLights(CBuffer1D<CLight> const& lights);

    /** Sets the volume data set to the specified buffer */
    static void setVolume(CVolumeBuffer const& volume);

    /** Sets the transfer function to the specified buffer */
    static void setTransfer(CTransferBuffer const& transfer);

    /** Sets the rendering parameters */
    static void setParameters(CRenderParams const& settings);

    /** Sets the kernel frame buffers for the active device */
    static void setFrameBuffers(
        CSampleBuffer2D const& sampleBuffer,
        CRandomBuffer2D const& rndSeeds0,
        CRandomBuffer2D const& rndSeeds1
        );
    
    /** Executes a single pass rendering kernel on the active device */
    static void execute(size_t xstart, size_t ystart,
                        size_t width,  size_t height);
};

}

// End definition
#endif // VSR_RENDER_KERNEL_H