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

// OpenGL+Glew Dependencies
#if defined(__APPLE__)
#  include <OpenGL/gl.h>
#else
#  include <GL/glew.h>
#  if defined(_WIN32)
#    include <GL/wglew.h>
#  endif
#  include <GL/gl.h>
#endif

// Include Header
#include "CudaRendererImpl.h"

// CUDA Kernel Parameters Headers
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "math_constants.h"

// Include Dependencies
#include "VoxLib/Core/CudaCommon.h"
#include "VoxLib/Core/Geometry.h"
#include "VoxLib/Core/Types.h"
#include "VoxLib/Error/CudaError.h"
#include "VoxLib/Scene/Camera.h"
#include "VoxLib/Scene/Film.h"
#include "VoxLib/Scene/Light.h"

// Device representations of scene components
#include "CudaRenderer/Core/CRandomGenerator.h"
#include "CudaRenderer/Scene/CCamera.h"
#include "CudaRenderer/Scene/CVolume.h"

// Optix SDK Headers
#include <optix.h>

// Optix Vector Math Operations
#include <optixu/optixu_math_namespace.h>

// API namespace
namespace vox
{

namespace {
namespace filescope {

} // namespace filescope
} // namespace anonymous
    
// --------------------------------------------------------------------
//  Prepares the renderer for use with the specified GPU device
// --------------------------------------------------------------------
std::shared_ptr<CudaRenderer> CudaRenderer::create()
{
    return std::make_shared<CudaRendererImpl>();
}

// --------------------------------------------------------------------
//  Prepares the renderer for use with the specified GPU device
// --------------------------------------------------------------------
CudaRendererImpl::CudaRendererImpl(int device) : 
    m_device(device),
    m_context(nullptr),
    m_filmHeight(0),
    m_filmWidth(0)
{
    m_ldrBuffer.init();
    m_hdrBuffer.init();
    m_rndSeeds0.init();
    m_rndSeeds1.init();
    m_lightBuffer.init();
}
    
// --------------------------------------------------------------------
//  Frees the GPU resources before shutdown
// --------------------------------------------------------------------
CudaRendererImpl::~CudaRendererImpl()
{
    m_ldrBuffer.reset();
    m_hdrBuffer.reset();
    m_rndSeeds0.reset();
    m_rndSeeds1.reset();
    m_lightBuffer.reset();
}

// --------------------------------------------------------------------
//  Prepares the renderer for the initialization of the render loop
// --------------------------------------------------------------------
void CudaRendererImpl::startup() 
{ 
    // :TODO: multiple devices
    cudaSetDevice(m_device); 
    
    // Ensure proper setup before optix context initialization
    cudaSetDeviceFlags(cudaDeviceMapHost | cudaDeviceLmemResizeToMax);

    // Reset the rand seed for seeding CUDA buffers
    srand(static_cast<unsigned int>(time(nullptr)));

    // Initialize the rendering context
    m_context = optix::Context::create();
    m_context->setRayTypeCount(2);
    m_context->setEntryPointCount(1);
    m_context->setStackSize(2800);

    // Initialize the context's individual Program components
    String const ptxFilepath = "C:/Users/Lucas/Documents/Projects/VoxSource/Binaries/x86/Source/CudaRenderer/CUDARenderer_generated_Programs.cu.ptx";
    m_context->setMissProgram         (0, m_context->createProgramFromPTXFile(ptxFilepath, "missProgram"));
    m_context->setRayGenerationProgram(0, m_context->createProgramFromPTXFile(ptxFilepath, "rayGenerationProgram"));
    m_context->setExceptionProgram    (0, m_context->createProgramFromPTXFile(ptxFilepath, "exceptionProgram"));

    // Global render settings
    m_context["radianceRayType"]->setUint(0);
    m_context["shadowRayType"]->setUint(1);
    m_context["sceneEpsilon"]->setFloat(0.001f);
    m_context["maxDepth"]->setUint(1);

    // Intialize a material to be applied to the translucent object
    std::string const transPtxFile = "C:/Users/Lucas/Documents/Projects/VoxSource/Binaries/x86/Source/CudaRenderer/CUDARenderer_generated_Translucent.cu.ptx";
    optix::Program transCH = m_context->createProgramFromPTXFile(transPtxFile, "closest_hit_radiance");
    optix::Program transAH = m_context->createProgramFromPTXFile(transPtxFile, "any_hit_shadow");

    // Translucent material for target object
    m_translucentMaterial = m_context->createMaterial();
    m_translucentMaterial->setClosestHitProgram(0, transCH);
    m_translucentMaterial->setAnyHitProgram(1, transAH);

    // Create a texture object for the volume
    auto buffer     = m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_BYTE, 1, 1, 1);
    auto texSampler = m_context->createTextureSampler();
    texSampler->setWrapMode(0, RT_WRAP_CLAMP_TO_EDGE);
    texSampler->setWrapMode(1, RT_WRAP_CLAMP_TO_EDGE);
    texSampler->setWrapMode(2, RT_WRAP_CLAMP_TO_EDGE);
    texSampler->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);
    texSampler->setIndexingMode(RT_TEXTURE_INDEX_ARRAY_INDEX);
    texSampler->setReadMode(RT_TEXTURE_READ_ELEMENT_TYPE);
    texSampler->setMaxAnisotropy(1.0f);
    texSampler->setMipLevelCount(1);
    texSampler->setArraySize(1);
    texSampler->setBuffer(0, 0, buffer);
    m_translucentMaterial["volumeTexture"]->setTextureSampler(texSampler);
}

// --------------------------------------------------------------------
//  Frees the rendering context after this render is terminated
// --------------------------------------------------------------------
void CudaRendererImpl::shutdown()
{
    // Destroy the OptiX context
    m_context->destroy();
}

// --------------------------------------------------------------------
//  Binds the specified scene components to the device memory
// --------------------------------------------------------------------
void CudaRendererImpl::syncScene(Scene const& scene)
{
    // Buffer size synchronization
    if (scene.film->isDirty())
    {
        // Establish the new film dimensions
        m_filmHeight = scene.film->height();
        m_filmWidth  = scene.film->width();

        // Resize the HDR sampling buffer 
        m_hdrBuffer.resize(m_filmWidth, m_filmHeight);
        m_context["sampleBuffer"]->setUserData(sizeof(m_hdrBuffer), &m_hdrBuffer);

        // Resize the CUDA RNG seed buffers 
        m_rndSeeds0.resize(m_filmWidth, m_filmHeight);
        m_rndSeeds1.resize(m_filmWidth, m_filmHeight);
        m_context["rndBuffer0"]->setUserData(sizeof(m_rndSeeds0), &m_rndSeeds0);
        m_context["rndBuffer1"]->setUserData(sizeof(m_rndSeeds1), &m_rndSeeds1);

        // Resize the LDR image buffer
        m_ldrBuffer.resize(m_filmWidth, m_filmHeight);
        m_context["imageBuffer"]->setUserData(sizeof(m_ldrBuffer), &m_ldrBuffer);

        // Resize host side framebuffer
        m_frameBuffer = std::make_shared<FrameBuffer>(m_filmWidth, m_filmHeight);
    }
    else
    {
        m_hdrBuffer.clear();
    }

    // Camera data synchronization (embeds film as well)
    if (scene.camera->isDirty() || scene.film->isDirty())
    {
        m_context["camera"]->setUserData(sizeof(CCamera), &CCamera(scene));
    }

    // Light buffer size synchronization
    if (scene.lightSet->isDirty()) 
    {
        auto nLights = scene.lightSet->lights().size();
        m_lightBuffer.resize(nLights);
    }

    // Light buffer data synchronization
    if (scene.lightSet->isContentDirty())
    {
        // Compose a linear array of light data for the device
        auto nLights = scene.lightSet->lights().size();
        std::vector<CLight> lightData; lightData.reserve(nLights);
        BOOST_FOREACH (auto & light, scene.lightSet->lights())
        {
            lightData.push_back( CLight(*light) );
        }

        // Copy and transfer the light buffer content to the device
        m_lightBuffer.write(lightData);
        m_context["lightBuffer"]->setUserData(sizeof(m_lightBuffer), &m_lightBuffer);
        auto i = m_lightBuffer.read();
    }

    // Volume data synchronization
    if (scene.volume->isDirty())
    {
        // Ensure the output buffer size matches the volume data
        auto const& extent = scene.volume->extent();
        auto buffer = m_translucentMaterial["volumeTexture"]->getTextureSampler()->getBuffer(0, 0);
        buffer->setSize(extent[0], extent[1], extent[2]);

        m_translucentMaterial["volumeExtent"]->setUint(extent[0], extent[1], extent[2]);
        m_translucentMaterial["rayStepSize"]->setFloat(1.0f);

        // Copy the volume data to a host mapped buffer
        void* map = buffer->map();
        auto size = extent[0]*extent[1]*extent[2];
        memcpy(map, scene.volume->data(), size); 
        buffer->unmap();
    }

    static bool init = false; // :DEBUG:
    if (!init)
    {
        init = true;

        // Scene geometry context changes
        if (true)
        {
            rebuildGeometry();
        }
    }

    // Validate/recompile the programs
    m_context->validate();
    m_context->compile();
}

// --------------------------------------------------------------------
//  Executes a series of rendering kernels and samples an image frame
// --------------------------------------------------------------------
void CudaRendererImpl::render()
{
    // Generate new seeds for the CUDA RNG seed buffer
    m_rndSeeds0.randomize(); m_rndSeeds1.randomize();

    // Perform a single pass using the OptiX tracer
    m_context->launch(0, m_filmWidth, m_filmHeight);

    // Read the data back to the host
    m_ldrBuffer.read(*m_frameBuffer);

    m_frameBuffer->wait(); // Await user lock release

    // Execute the user defined callback routine
    boost::mutex::scoped_lock lock(m_mutex);
    if (m_callback) m_callback(m_frameBuffer);
}

// --------------------------------------------------------------------
//  Rebuilds the device geometry structures 
// --------------------------------------------------------------------
void CudaRendererImpl::rebuildGeometry()
{
    std::string const spherePtxFile = "C:/Users/Lucas/Documents/Projects/VoxSource/Binaries/x86/Source/CudaRenderer/CUDARenderer_generated_Sphere.cu.ptx";
    std::string const pgramPtxFile  = "C:/Users/Lucas/Documents/Projects/VoxSource/Binaries/x86/Source/CudaRenderer/CUDARenderer_generated_Parallelogram.cu.ptx";
    std::string const checkPtxFile  = "C:/Users/Lucas/Documents/Projects/VoxSource/Binaries/x86/Source/CudaRenderer/CUDARenderer_generated_Checker.cu.ptx";

    // Construct floor geometry as a plane object
    optix::Geometry translucentObj = m_context->createGeometry();
    translucentObj->setPrimitiveCount(1u);
    translucentObj->setBoundingBoxProgram(m_context->createProgramFromPTXFile(spherePtxFile, "bounds"));
    translucentObj->setIntersectionProgram(m_context->createProgramFromPTXFile(spherePtxFile, "intersect"));
    translucentObj["sphere"]->setFloat(48.0f, 48.0f, 48.0f, 32.0f);

    // Construct floor geometry as a plane object
    optix::Geometry parallelogram = m_context->createGeometry();
    parallelogram->setPrimitiveCount(1u);
    parallelogram->setBoundingBoxProgram(m_context->createProgramFromPTXFile(pgramPtxFile, "bounds"));
    parallelogram->setIntersectionProgram(m_context->createProgramFromPTXFile(pgramPtxFile, "intersect"));
    float3 anchor = optix::make_float3(0.0f, 0.0f, 0.0f);
    float3 v1 = optix::make_float3(0.0f, 0.0f, 128.0f);
    float3 v2 = optix::make_float3(128.0f, 0.0f, 0.0f);
    float3 normal = optix::cross(v1, v2);
    normal = optix::normalize(normal);
    float d = optix::dot(normal, anchor);
    v1 *= 1.0f / optix::dot(v1, v1);
    v2 *= 1.0f / optix::dot(v2, v2);
    float4 plane = optix::make_float4(normal, d);
    parallelogram["plane"]->setFloat(plane);
    parallelogram["v1"]->setFloat(v1);
    parallelogram["v2"]->setFloat(v2);
    parallelogram["anchor"]->setFloat(anchor);

    // Checkered material for flooring
    optix::Program checkCH = m_context->createProgramFromPTXFile(checkPtxFile, "closest_hit_radiance");
    optix::Program checkAH = m_context->createProgramFromPTXFile(checkPtxFile, "any_hit_shadow");
    optix::Material floor_matl = m_context->createMaterial();
    floor_matl->setClosestHitProgram(0, checkCH);
    floor_matl->setAnyHitProgram(1, checkAH);

    floor_matl["Kd1"]->setFloat(0.8f, 0.3f, 0.15f);
    floor_matl["Ka1"]->setFloat(0.8f, 0.3f, 0.15f);
    floor_matl["Ks1"]->setFloat(0.0f, 0.0f, 0.0f);
    floor_matl["Kd2"]->setFloat(0.9f, 0.85f, 0.05f);
    floor_matl["Ka2"]->setFloat(0.9f, 0.85f, 0.05f);
    floor_matl["Ks2"]->setFloat(0.0f, 0.0f, 0.0f);
    floor_matl["inv_checker_size"]->setFloat(32.0f, 16.0f, 1.0f);
    floor_matl["phong_exp1"]->setFloat(0.0f);
    floor_matl["phong_exp2"]->setFloat(0.0f);
    floor_matl["reflectivity1"]->setFloat(0.0f, 0.0f, 0.0f);
    floor_matl["reflectivity2"]->setFloat(0.0f, 0.0f, 0.0f);

    // Create GIs for each piece of geometry
    std::vector<optix::GeometryInstance> gis;
    gis.push_back( m_context->createGeometryInstance( translucentObj, &m_translucentMaterial, &m_translucentMaterial+1 ) );
    gis.push_back( m_context->createGeometryInstance( parallelogram, &floor_matl,  &floor_matl+1 ) );

    // Construct the scene geometry group hierarchy 
    optix::GeometryGroup geometrygroup = m_context->createGeometryGroup();
    geometrygroup->setChildCount( gis.size() );
    geometrygroup->setChild(0, gis[0]);
    geometrygroup->setChild(1, gis[1]);
    geometrygroup->setAcceleration(m_context->createAcceleration("NoAccel","NoAccel"));
    m_context["geometryRoot"]->set( geometrygroup );
}

}