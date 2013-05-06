/* ===========================================================================

	Project: VoxRender - Scene

	Description: Defines the Scene class used by the Renderer

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
#include "Scene.h"

// Include Dependencies
#include "VoxLib/Core/Common.h"
#include "VoxLib/Core/Format.h"
#include "VoxLib/Core/Logging.h"
#include "VoxLib/Error/Error.h"
#include "VoxLib/Error/ErrorCodes.h"

// API namespace
namespace vox
{

namespace {
namespace filescope {

    static std::map<String, SceneImporter> importers;   // Registered import modules
    static std::map<String, SceneExporter> exporters;   // Registered export modules    
    static boost::shared_mutex             moduleMutex; // Module access mutex for read-write locks

    // --------------------------------------------------------------------
    //  Helper function for issuing warning for missing scene data
    // --------------------------------------------------------------------
    void issueWarning(char const* com)
    {
        Logger::addEntry(Severity_Warning, Error_MissingData, VOX_LOG_CATEGORY,
                         format("%1% handle is not present in scene context", com),
                         __FILE__, __LINE__);
    }

} // namespace filescope
} // namespace anonymous

// --------------------------------------------------------------------
//  Registered import modules accessor :TODO: privatize, used for testing
// --------------------------------------------------------------------
std::map<String, SceneImporter> const& Scene::importers() 
{ 
    return filescope::importers; 
}

// --------------------------------------------------------------------
//  Registered export modules accessor
// --------------------------------------------------------------------
std::map<String, SceneExporter> const& Scene::exporters() 
{ 
    return filescope::exporters; 
}

// --------------------------------------------------------------------
//  Registers a new resource import module
// --------------------------------------------------------------------
void Scene::registerImportModule(String const& extension, SceneImporter importer)
{ 
    // Acquire a read-lock on the modules for thread safety support
    boost::unique_lock<decltype(filescope::moduleMutex)> lock(filescope::moduleMutex);

    filescope::importers[extension] = importer; 
}

// --------------------------------------------------------------------
//  Registers a new resource export module
// --------------------------------------------------------------------
void Scene::registerExportModule(String const& extension, SceneExporter exporter)
{ 
    // Acquire a read-lock on the modules for thread safety support
    boost::unique_lock<decltype(filescope::moduleMutex)> lock(filescope::moduleMutex);

    filescope::exporters[extension] = exporter; 
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
        return importer->second(data, options);
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
        exporter->second(data, options, *this);
    }
    else
    {
        throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY, 
                    "No export module found", Error_BadToken);
    }
}

// --------------------------------------------------------------------
//  Logs warning message for any missing scene components
// --------------------------------------------------------------------
void Scene::issueWarningsForMissingHandles() const
{
    if (!camera)   filescope::issueWarning("Camera");
    if (!volume)   filescope::issueWarning("Volume");
    if (!lightSet) filescope::issueWarning("LightSet");
    if (!transfer) filescope::issueWarning("Transfer Function");
}

} // namespace vox