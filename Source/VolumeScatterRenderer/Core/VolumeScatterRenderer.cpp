/* ===========================================================================

	Project: GPU based Volume Scatter Renderer
    
	Description: Implements CUDA based volume rendering

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

// :TODO: Investigate use of mallocHost for Scene components

// Include Header
#include "VolumeScatterRenderer.h"

// CUDA Kernel Parameters Headers
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "math_constants.h"

// Include Dependencies
#include "VoxLib/Core/CudaCommon.h"
#include "VoxLib/Core/Geometry.h"
#include "VoxLib/Core/Logging.h"
#include "VoxLib/Core/Types.h"
#include "VoxLib/Error/CudaError.h"
#include "VoxLib/Scene/Camera.h"
#include "VoxLib/Scene/Film.h"
#include "VoxLib/Scene/Light.h"
#include "VoxLib/Scene/Transfer.h"
#include "VoxLib/Scene/Volume.h"
#include "VoxLib/Scene/PrimGroup.h"

// Device representations of scene components
#include "VolumeScatterRenderer/Core/CBuffer.h"
#include "VolumeScatterRenderer/Core/CRandomBuffer.h"
#include "VolumeScatterRenderer/Core/CRandomGenerator.h"
#include "VolumeScatterRenderer/Core/CSampleBuffer.h"
#include "VolumeScatterRenderer/Scene/CCamera.h"
#include "VolumeScatterRenderer/Scene/CClipGeometry.h"
#include "VolumeScatterRenderer/Scene/CLight.h"
#include "VolumeScatterRenderer/Scene/CTransferBuffer.h"
#include "VolumeScatterRenderer/Scene/CVolumeBuffer.h"
#include "VolumeScatterRenderer/Scene/CRenderParams.h"

// Interface for accessing device render kernels 
#include "VolumeScatterRenderer/Kernels/RenderKernel.h"
#include "VolumeScatterRenderer/Kernels/TonemapKernel.h"

// API namespace
namespace vox
{

namespace {
namespace filescope {

} // namespace filescope
} // namespace anonymous

// --------------------------------------------------------------------
//  Implements the GPU Renderer of the VolumeScatter Renderers 
// --------------------------------------------------------------------
class VolumeScatterRendererImpl : public VolumeScatterRenderer
{
public:
    // --------------------------------------------------------------------
    //  Prepares the renderer for use with the specified GPU device
    // --------------------------------------------------------------------
    VolumeScatterRendererImpl()
    {
        m_ldrBuffer.init();
        m_hdrBuffer.init();
        m_rndSeeds0.init();
        m_rndSeeds1.init();
        m_lightBuffer.init();
        m_transferBuffer.init();
        m_volumeBuffer.init();
    }
    
    // --------------------------------------------------------------------
    //  Frees the GPU resources before shutdown
    // --------------------------------------------------------------------
    ~VolumeScatterRendererImpl()
    {
        try
        {
            shutdown();
        } 
        catch(Error & error) 
        {
            error.message = "Error encountered during VolumeScatterRendererGPU shutdown >> " + error.message;
            Logger::addEntry(error, Severity_Warning); // Probably not necessarily critical but could cause problems
        }
    }

    // --------------------------------------------------------------------
    //  Prepares the renderer for the initialization of the render loop
    // --------------------------------------------------------------------
    void startup() 
    { 
        // Reset the rand seed for seeding CUDA buffers
        srand(static_cast<unsigned int>(time(nullptr)));
    }

    // --------------------------------------------------------------------
    //  Binds the specified scene components to the device memory
    // --------------------------------------------------------------------
    void syncScene(Scene const& scene)
    {                
        // Buffer size synchronization
        if (scene.camera->isFilmDirty())
        {
            size_t filmHeight = scene.camera->filmHeight();
            size_t filmWidth  = scene.camera->filmWidth();

            //BOOST_FOREACH(auto & device, m_devices)
            {
                // Resize the device frame buffers
                m_hdrBuffer.resize(filmWidth, filmHeight);
                m_rndSeeds0.resize(filmWidth, filmHeight);
                m_rndSeeds1.resize(filmWidth, filmHeight);
                m_ldrBuffer.resize(filmWidth, filmHeight);

                // :DEBUG:
                RenderKernel::setFrameBuffers(m_hdrBuffer,
                                              m_rndSeeds0,
                                              m_rndSeeds1);
            }

            // Allocate/Resize the host side LDR framebuffer
            if (m_frameBuffer) 
            {
                m_frameBuffer->wait(); m_frameBuffer->resize(filmWidth, filmHeight);
            }
            else m_frameBuffer.reset(new FrameBuffer(filmWidth, filmHeight));
        }
        else
        {
            m_hdrBuffer.clear();
        }

        // Volume data synchronization
        if (scene.volume->isDirty())
        {
            m_volumeBuffer.setVolume(scene.volume);

            RenderKernel::setVolume(m_volumeBuffer);
        }

        // Transfer function data synchronization
        if (scene.transfer)
        {
            if (scene.transfer->isDirty())
            {
                scene.transfer->generateMap(scene.transferMap);
                m_transferBuffer.setTransfer(scene.transferMap);
                RenderKernel::setTransfer(m_transferBuffer);
            }
        }
        else if (scene.transferMap->isDirty())
        {
            m_transferBuffer.setTransfer(scene.transferMap);
            RenderKernel::setTransfer(m_transferBuffer);
        }

        // Render settings synchronization
        if (scene.parameters->isDirty()) 
        {
            RenderKernel::setParameters(CRenderParams(scene.parameters));
        }

        // Camera data synchronization 
        if (scene.camera->isDirty())
        {
            RenderKernel::setCamera(CCamera(scene.camera));

            // :TODO: Move to seperate dirty flag and don't reset film
            m_exposure = scene.camera->exposure();
        }

        // Clipping geometry synchronization
        if (scene.clipGeometry->isDirty())
        {
            RenderKernel::setClipRoot(CClipGeometry::create(scene.clipGeometry));
        }

        // Light data synchronization
        if (scene.lightSet->isDirty())
        {
            // Construct an array of CUDA light objects
            auto lights = scene.lightSet->lights();
            std::vector<CLight> clights;
            BOOST_FOREACH(auto & light, lights)
            {
                clights.push_back( CLight(*light) );
            }
            
            m_lightBuffer.write(clights);

            RenderKernel::setLights(m_lightBuffer, scene.lightSet->ambientLight());
        }
    }

    // --------------------------------------------------------------------
    //  Executes a series of rendering kernels and samples an image frame
    // --------------------------------------------------------------------
    void render()
    {
        // Generate new seeds for the CUDA RNG seed buffer
        m_rndSeeds0.randomize(); m_rndSeeds1.randomize();

        // Execute one cycle of the device rendering kernel
        RenderKernel::execute(0, 0, m_hdrBuffer.width(), m_hdrBuffer.height());

        // Perform tonemapping on the HDR image buffer
        TonemapKernel::execute(m_hdrBuffer, m_ldrBuffer, m_exposure);

        m_frameBuffer->wait(); // Await user lock release

        // Read the data back to the host
        m_ldrBuffer.read(*m_frameBuffer);

        // Execute the user defined callback routine
        boost::mutex::scoped_lock lock(m_mutex);
        if (m_callback) m_callback(m_frameBuffer);
    }

    // --------------------------------------------------------------------
    //  Sets the post render event callback function for this renderer
    // --------------------------------------------------------------------
    virtual void setRenderEventCallback(RenderCallback callback)
    {
        boost::mutex::scoped_lock lock(m_mutex);

        m_callback = callback;
    }
    
    // --------------------------------------------------------------------
    //  Exports the scene data to the output resource :TODO:
    // --------------------------------------------------------------------
    virtual void backupIpr(std::ostream & out) { } 
    
    // --------------------------------------------------------------------
    //  Merges the input image buffer with the internal one :TODO:
    // --------------------------------------------------------------------
    virtual void pushIpr(IprImage const& ipr, size_t const& samples) { }

    // --------------------------------------------------------------------
    //  Pulls the current in-progress-render buffer then clears it :TODO:
    // --------------------------------------------------------------------
    virtual void pullIpr(IprImage & img, size_t & samples) { }

    // --------------------------------------------------------------------
    //  Terminates rendering operations and clears device data buffers
    // --------------------------------------------------------------------
	virtual void shutdown()
	{
		m_ldrBuffer.reset();
		m_hdrBuffer.reset();
		m_rndSeeds0.reset();
		m_rndSeeds1.reset();
		m_lightBuffer.reset();
		m_transferBuffer.reset();
		m_volumeBuffer.reset();
		m_frameBuffer.reset();
	}

    // --------------------------------------------------------------------
    //  Returns the time for the last call to the render kernel
    // --------------------------------------------------------------------
    virtual float renderTime()
    {
        return RenderKernel::getTime();
    }
    
    // --------------------------------------------------------------------
    //  Returns the time for the last call to the tonemapping kernel
    // --------------------------------------------------------------------
    virtual float tonemapTime()
    {
        return TonemapKernel::getTime();
    }

private:
    std::vector<int> m_devices; /// Authorized Device IDs

    CImgBuffer2D<ColorRgbaLdr> m_ldrBuffer;    ///< LDR post-processed image

    CSampleBuffer2D m_hdrBuffer;    ///< HDR raw sample data buffer
    CRandomBuffer2D m_rndSeeds0;    ///< Seed buffer for CUDA RNG
    CRandomBuffer2D m_rndSeeds1;    ///< Seed buffer for CUDA RNG

    CBuffer1D<CLight> m_lightBuffer;    ///< Array of scene lights
    CVolumeBuffer     m_volumeBuffer;   ///< Volume data buffer
    CTransferBuffer   m_transferBuffer; ///< Transfer function data buffer

    float m_exposure; ///< Exposure factor

    RenderCallback m_callback; ///< User defined render callback
    boost::mutex   m_mutex;    ///< Mutex for callback modification

    std::shared_ptr<FrameBuffer> m_frameBuffer;  ///< Host side framebuffer
};

// --------------------------------------------------------------------
//  Creates an instance of the VolumeScatterRenderer's implementation
// --------------------------------------------------------------------
std::shared_ptr<VolumeScatterRenderer> VolumeScatterRenderer::create()
{
    return std::make_shared<VolumeScatterRendererImpl>();
}

} // namespace vox