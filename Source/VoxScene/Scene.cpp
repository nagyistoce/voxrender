/* ===========================================================================

	Project: VoxRender - Scene

	Description: Defines the Scene class used by the Renderer

    Copyright (C) 2012-2013 Lucas Sherman

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
#include "Scene.h"

// Include Dependencies
#include "VoxLib/Core/Common.h"
#include "VoxLib/Core/Format.h"
#include "VoxLib/Core/Logging.h"
#include "VoxLib/Error/Error.h"
#include "VoxLib/Error/ErrorCodes.h"
#include "VoxScene/Transfer.h"
#include "VoxScene/TransferMap.h"
#include "VoxScene/Camera.h"
#include "VoxScene/Light.h"
#include "VoxScene/PrimGroup.h"
#include "VoxScene/RenderParams.h"
#include "VoxScene/Volume.h"

// API namespace
namespace vox
{

namespace {
namespace filescope {

    static std::map<String, std::shared_ptr<SceneImporter>> importers;   // Registered import modules
    static std::map<String, std::shared_ptr<SceneExporter>> exporters;   // Registered export modules 

    static boost::shared_mutex moduleMutex; // Module access mutex for read-write locks

} // namespace filescope
} // namespace anonymous

// --------------------------------------------------------------------
//  Resets the scene element pointers
// --------------------------------------------------------------------
void Scene::reset()
{
    camera.reset();
    transfer.reset(); 
    lightSet.reset(); 
    volume.reset(); 
    clipGeometry.reset();
    parameters.reset();
}

// --------------------------------------------------------------------
//  Registers a new resource import module
// --------------------------------------------------------------------
void Scene::registerImportModule(String const& extension, std::shared_ptr<SceneImporter> importer)
{ 
    // Acquire a read-lock on the modules for thread safety support
    boost::unique_lock<decltype(filescope::moduleMutex)> lock(filescope::moduleMutex);
    
    if (!importer) throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY, 
        "Attempted to register empty import module", Error_NotAllowed);

    filescope::importers[extension] = importer; 
}

// --------------------------------------------------------------------
//  Registers a new resource export module
// --------------------------------------------------------------------
void Scene::registerExportModule(String const& extension, std::shared_ptr<SceneExporter> exporter)
{ 
    // Acquire a read-lock on the modules for thread safety support
    boost::unique_lock<decltype(filescope::moduleMutex)> lock(filescope::moduleMutex);

    if (!exporter) throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY, 
        "Attempted to register empty export module", Error_NotAllowed);

    filescope::exporters[extension] = exporter; 
}

// --------------------------------------------------------------------
//  Removes a scene import module
// --------------------------------------------------------------------
void Scene::removeImportModule(std::shared_ptr<SceneImporter> importer, String const& extension)
{
    // Acquire a read-lock on the modules for thread safety support
    boost::unique_lock<decltype(filescope::moduleMutex)> lock(filescope::moduleMutex);
    
    if (extension.empty())
    {
        auto iter = filescope::importers.begin();
        while (iter != filescope::importers.end())
        {
            if (iter->second == importer)
            {
                auto old = iter; ++iter;
                filescope::importers.erase(old);
            }
            else
            {
                ++iter;
            }
        }
    }
    else
    {
        auto iter = filescope::importers.find(extension);
        if (iter != filescope::importers.end()) 
            filescope::importers.erase(iter);
    }
}

// --------------------------------------------------------------------
//  Removes a scene export module
// --------------------------------------------------------------------
void Scene::removeExportModule(std::shared_ptr<SceneExporter> exporter, String const& extension)
{
    // Acquire a read-lock on the modules for thread safety support
    boost::unique_lock<decltype(filescope::moduleMutex)> lock(filescope::moduleMutex);
    
    if (extension.empty())
    {
        auto iter = filescope::exporters.begin();
        while (iter != filescope::exporters.end())
        {
            if (iter->second == exporter)
            {
                auto old = iter; ++iter;
                filescope::exporters.erase(old);
            }
            else
            {
                ++iter;
            }
        }
    }
    else
    {
        auto iter = filescope::exporters.find(extension);
        if (iter != filescope::exporters.end()) 
            filescope::exporters.erase(iter);
    }
}

// --------------------------------------------------------------------
//  Imports a scene using a matching registered importer
// --------------------------------------------------------------------
Scene Scene::imprt(ResourceIStream & data, OptionSet const& options, String const& extension)
{
    // Acquire a read-lock on the modules for thread safety support
    boost::shared_lock<decltype(filescope::moduleMutex)> lock(filescope::moduleMutex);

    String type = extension.empty() ? data.identifier().extractFileExtension() : extension;

	// Execute the register import module
    auto importer = filescope::importers.find(type);
    if (importer != filescope::importers.end())
    {
        return importer->second->importer(data, options);
    }

    throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY, 
                "No import module found", Error_BadToken);
}

// --------------------------------------------------------------------
//  Exports a scene using a matching registered exporter
// --------------------------------------------------------------------
void Scene::exprt(ResourceOStream & data, OptionSet const& options, String const& extension) const
{
    // Acquire a read-lock on the modules for thread safety support
    boost::shared_lock<decltype(filescope::moduleMutex)> lock(filescope::moduleMutex);

    String type = extension.empty() ? data.identifier().extractFileExtension() : extension;

	// Execute the register import module
    auto exporter = filescope::exporters.find(type);
    if (exporter != filescope::exporters.end())
    {
        exporter->second->exporter(data, options, *this);
    }
    else
    {
        throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY, 
                    "No export module found", Error_BadToken);
    }
}

// --------------------------------------------------------------------
//  Clones a scene, referencing the volume and copying the other comps
// --------------------------------------------------------------------
bool Scene::isDirty() const
{
    return (camera && camera->isDirty())           ||
           (lightSet && lightSet->isDirty())       ||
           (volume && volume->isDirty())           ||
           (parameters && parameters->isDirty())   ||
           (transfer && transfer->isDirty())       ||
           (transferMap && transferMap->isDirty()) ||
           (clipGeometry && clipGeometry->isDirty());
}

// --------------------------------------------------------------------
//  Constructs a keyframe for the current state of this scene
// --------------------------------------------------------------------
KeyFrame Scene::generateKeyFrame()
{
    Scene scene;
    clone(scene);
    return scene;
}

// --------------------------------------------------------------------
//  Clones a scene, referencing the volume and copying the other comps
// --------------------------------------------------------------------
void Scene::clone(Scene & scene) const
{
    // :TODO:
    scene.clipGeometry = clipGeometry;
    scene.animator = nullptr;
    scene.transfer = transfer;
    //

    if (volume)
    {
        if (!scene.volume) scene.volume = Volume::create();
        volume->clone(*scene.volume.get());
    }

    if (lightSet)
    {
        if (!scene.lightSet) scene.lightSet = LightSet::create();
        lightSet->clone(*scene.lightSet.get());
    }

    if (camera)
    {
        if (!scene.camera) scene.camera = Camera::create();
        camera->clone(*scene.camera.get());
    }
    else scene.camera.reset();
    
    if (parameters)
    {
        if (!scene.parameters) scene.parameters = RenderParams::create();
        parameters->clone(*scene.parameters.get());
    }
    else scene.parameters.reset();

    if (transfer)
    {
    }
    else if (transferMap)
    {
    }
    else scene.transferMap.reset();
}

// --------------------------------------------------------------------
//  Returns true if the scene is a valid render scene
// --------------------------------------------------------------------
bool Scene::isValid() const
{
    return camera && volume && lightSet && transfer && clipGeometry && parameters && (transfer || transferMap);
}

} // namespace vox