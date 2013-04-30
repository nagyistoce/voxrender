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
#include "VoxLib/Scene/Transfer.h"

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
    String const ptxFilepath = "C:/Users/Lucas/Documents/Projects/voxrender/trunk/Binaries/x86/Source/CudaRenderer/CUDARenderer_generated_Programs.cu.ptx";
    m_context->setMissProgram         (0, m_context->createProgramFromPTXFile(ptxFilepath, "missProgram"));
    m_context->setRayGenerationProgram(0, m_context->createProgramFromPTXFile(ptxFilepath, "rayGenerationProgram"));
    m_context->setExceptionProgram    (0, m_context->createProgramFromPTXFile(ptxFilepath, "exceptionProgram"));

    // Global render settings
    m_context["radianceRayType"]->setUint(0);
    m_context["shadowRayType"]->setUint(1);
    m_context["sceneEpsilon"]->setFloat(0.001f);
    m_context["maxDepth"]->setUint(1);

    // Intialize a material to be applied to the translucent object
    std::string const transPtxFile = "C:/Users/Lucas/Documents/Projects/voxrender/trunk/Binaries/x86/Source/CudaRenderer/CUDARenderer_generated_Translucent.cu.ptx";
    optix::Program transCH = m_context->createProgramFromPTXFile(transPtxFile, "closest_hit_radiance");
    optix::Program transAH = m_context->createProgramFromPTXFile(transPtxFile, "any_hit_shadow");

    // Translucent material for target object
    m_translucentMaterial = m_context->createMaterial();
    m_translucentMaterial->setClosestHitProgram(0, transCH);
    m_translucentMaterial->setAnyHitProgram(1, transAH);

    // Create a texture object for the volume
    auto buffer     = m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, 1, 1, 1);
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

    // Create a texture object for the transfer function
    auto buffer2     = m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_BYTE4, 1);
    auto texSampler2 = m_context->createTextureSampler();
    texSampler2->setWrapMode(0, RT_WRAP_CLAMP_TO_EDGE);
    texSampler2->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);
    texSampler2->setIndexingMode(RT_TEXTURE_INDEX_ARRAY_INDEX);
    texSampler2->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
    texSampler2->setMaxAnisotropy(1.0f);
    texSampler2->setMipLevelCount(1);
    texSampler2->setArraySize(1);
    texSampler2->setBuffer(0, 0, buffer2);
    m_translucentMaterial["transferTexture"]->setTextureSampler(texSampler2);
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
    /*
    // Volume data synchronization
    if (scene.volume->isDirty())
    {
        // Ensure the output buffer size matches the volume data
        auto const& extent = scene.volume->extent();
        auto buffer = m_translucentMaterial["volumeTexture"]->getTextureSampler()->getBuffer(0, 0);
        buffer->setSize(extent[0], extent[1], extent[2]);

        m_translucentMaterial["volumeExtent"]->setUint(extent[0], extent[1], extent[2]);
        m_translucentMaterial["rayStepSize"]->setFloat(0.25f);
        m_translucentMaterial["specularFactor"]->setFloat(0.2f);

        // Copy the volume data to a host mapped buffer
        void* map = buffer->map();
        auto size = extent[0]*extent[1]*extent[2];
        memcpy(map, scene.volume->data(), size); 
        buffer->unmap();
    }
    */
    {
        /*
        // Construct the transfer function
        auto buffer = m_translucentMaterial["transferTexture"]->getTextureSampler()->getBuffer(0, 0);

        buffer->setSize(256);
        void* map = buffer->map();
        memcpy(map, scene.transfer->debugMap(), 256*sizeof(uchar4)); 
        //memset(map, -1, 256*sizeof(uchar4));
        buffer->unmap();
        */
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
    std::string const spherePtxFile = "C:/Users/Lucas/Documents/Projects/voxrender/trunk/Binaries/x86/Source/CudaRenderer/CUDARenderer_generated_Sphere.cu.ptx";
    std::string const pgramPtxFile  = "C:/Users/Lucas/Documents/Projects/voxrender/trunk/Binaries/x86/Source/CudaRenderer/CUDARenderer_generated_Parallelogram.cu.ptx";
    std::string const checkPtxFile  = "C:/Users/Lucas/Documents/Projects/voxrender/trunk/Binaries/x86/Source/CudaRenderer/CUDARenderer_generated_Checker.cu.ptx";
    std::string const tmeshPtxFile  = "C:/Users/Lucas/Documents/Projects/voxrender/trunk/Binaries/x86/Source/CudaRenderer/CUDARenderer_generated_TriangleMeshSmall.cu.ptx";

    // Construct translucent geometry mesh
    optix::Geometry translucentObj = m_context->createGeometry();
    translucentObj->setBoundingBoxProgram(m_context->createProgramFromPTXFile(tmeshPtxFile, "mesh_bounds"));
    translucentObj->setIntersectionProgram(m_context->createProgramFromPTXFile(tmeshPtxFile, "mesh_intersect"));

    {
        // :TODO: Photon map structure
        std::ifstream pstream("C:/Users/Lucas/Documents/Projects/voxrender/trunk/models/out.tex");

        Vector<int,3>   dimension;
        float           resolution;
        Vector<float,3> origin;

        pstream.read((char*)&dimension, sizeof(dimension));
        pstream.read((char*)&resolution, sizeof(resolution));
        pstream.read((char*)&origin, sizeof(origin));

        resolution *= 50.0f;

        std::vector<float4> data( dimension.fold(1, mul<int>) );
        for (size_t z = 0; z < dimension[2]; z++)
        for (size_t y = 0; y < dimension[1]; y++)
        for (size_t x = 0; x < dimension[0]; x++)
        {
            size_t i = x + y * dimension[0] + z * dimension[0] * dimension[1];

            pstream.read((char*)&data[i], sizeof(float3));
            
            if (z < dimension[2]/4 || z > dimension[2]/4 + dimension[2]/2)
            {
                data[i].x *= 0;
                data[i].y *= 0;
                data[i].z *= 0;
            }
            else
            {
                data[i].x = 1000.0f;
                data[i].y = 1000.0f;
                data[i].z = 1000.0f;
            }
        }
       
        pstream.close();

        float rayStepSize = 0.0001f;
        float3 transmission = expf( - rayStepSize * optix::make_float3(2190.1, 2620.1f, 3000.1f) );

        m_translucentMaterial["extent"]->setUint(dimension[0], dimension[1], dimension[2]);
        m_translucentMaterial["rayStepSize"]->setFloat(rayStepSize);
        m_translucentMaterial["specularFactor"]->setFloat(0.0f);
        m_translucentMaterial["invSpacing"]->setFloat(1.0f/resolution, 1.0f/resolution, 1.0f/resolution);
        m_translucentMaterial["anchor"]->setFloat(origin[0], origin[1], origin[2]);

        optix::Buffer pbuffer = m_translucentMaterial["volumeTexture"]->getTextureSampler()->getBuffer(0, 0);
        pbuffer->setSize(dimension[0], dimension[1], dimension[2]);
        auto pbufmap = pbuffer->map();
        memcpy(pbufmap, &data[0], data.size()*sizeof(float4));
        m_translucentMaterial["transmission"]->setFloat(transmission);
        pbuffer->unmap();
    }

    // :TODO: Mesh object
    std::ifstream filestream("C:/Users/Lucas/Documents/Projects/voxrender/trunk/models/dragon.pbrt");

    // Read vertices
    std::vector<float3> vertices;
    { std::string delim; filestream >> delim; }
    while (filestream)
    {
        std::string x, y, z;
        filestream >> x; if (x == "]") break;
        filestream >> y; 
        filestream >> z;
        vertices.push_back(optix::make_float3(
            boost::lexical_cast<float>(x),
            boost::lexical_cast<float>(y),
            boost::lexical_cast<float>(z)
            ));
    }

    optix::Buffer vbuffer = m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, vertices.size()); 
    auto vbufmap = vbuffer->map();
    memcpy(vbufmap, &vertices[0], vertices.size()*sizeof(float3));
    translucentObj["vertex_buffer"]->setBuffer(vbuffer);
    vbuffer->unmap();

    // Read indices
    std::vector<int3> indices;
    { std::string delim; filestream >> delim; }
    while (filestream)
    {
        std::string x, y, z;
        filestream >> x; if (x == "]") break;
        filestream >> y; 
        filestream >> z;
        indices.push_back(optix::make_int3(
            boost::lexical_cast<int>(x),
            boost::lexical_cast<int>(y),
            boost::lexical_cast<int>(z)
            ));
    }
    
    translucentObj->setPrimitiveCount(indices.size());
    optix::Buffer ibuffer = m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_INT3, indices.size()); 
    auto ibufmap = ibuffer->map();
    memcpy(ibufmap, &indices[0], indices.size()*sizeof(int3));
    ibuffer->unmap();
    translucentObj["vindex_buffer"]->setBuffer(ibuffer);

    // Read normals
    std::vector<float3> normals;
    { std::string delim; filestream >> delim; }
    while (filestream)
    {
        std::string x, y, z;
        filestream >> x; if (x == "]") break;
        filestream >> y; 
        filestream >> z;
        normals.push_back(optix::make_float3(
            boost::lexical_cast<float>(x),
            boost::lexical_cast<float>(y),
            boost::lexical_cast<float>(z)
            ));
    }
    
    optix::Buffer nbuffer = m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, normals.size()); 
    auto nbufmap = nbuffer->map();
    memcpy(nbufmap, &normals[0], normals.size()*sizeof(float3));
    nbuffer->unmap();
    translucentObj["normal_buffer"]->setBuffer(nbuffer);

    filestream.close();

    // Construct floor geometry as a plane object
    optix::Geometry parallelogram = m_context->createGeometry();
    parallelogram->setPrimitiveCount(1u);
    parallelogram->setBoundingBoxProgram(m_context->createProgramFromPTXFile(pgramPtxFile, "bounds"));
    parallelogram->setIntersectionProgram(m_context->createProgramFromPTXFile(pgramPtxFile, "intersect"));
    float3 anchor = optix::make_float3(-64.0f, 0.0f, -64.0f);
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
    geometrygroup->setAcceleration(m_context->createAcceleration("Bvh","Bvh"));
    m_context["geometryRoot"]->set( geometrygroup );
}

}