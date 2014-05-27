/* ===========================================================================

    Project: Vox Scene Import Module
    
	Description: Defines a VoxScene import module for .raw format volumes

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

// Include Header
#include "Plugin.h"

// Include Dependencies
#include "VoxSceneImporter/Common.h"
#include "VoxSceneImporter/VoxSceneImporter.h"
#include "VoxLib/Core/Logging.h"
#include "VoxLib/Plugin/PluginManager.h"

using namespace vox;

namespace {
namespace filescope {

    std::shared_ptr<VoxSceneFile> eximXml;
    std::shared_ptr<VoxSceneFile> eximJson;
    std::shared_ptr<void> handle;

} // namespace filescope
} // namespace anonymous

// --------------------------------------------------------------------
//  Deletes the specified file or directory 
// --------------------------------------------------------------------
void initPlugin() 
{ 
    VOX_LOG_INFO(VSI_LOG_CATEGORY, "Loading the 'Vox.Vox Scene ExIm' plugin");
    
    filescope::handle = PluginManager::instance().acquirePluginHandle();
}

// --------------------------------------------------------------------
//  Deletes the specified file or directory 
// --------------------------------------------------------------------
void freePlugin() 
{ 
    VOX_LOG_INFO(VSI_LOG_CATEGORY, "Unloading the 'Vox.Vox Scene ExIm' plugin");
}

// --------------------------------------------------------------------
//  Returns the dot delimited version string for this build
// --------------------------------------------------------------------
char const* version() { return VSI_VERSION_STRING; }

// --------------------------------------------------------------------
//  Returns a reference URL for the plugin
// --------------------------------------------------------------------
char const* referenceUrl() { return "http://code.google.com/p/voxrender/"; }

// --------------------------------------------------------------------
//  Returns the minimum compatible version of the plugin API
// --------------------------------------------------------------------
char const* apiVersionMin() { return VSI_API_VERSION_MIN_STR; }

// --------------------------------------------------------------------
//  Returns the minimum compatible version of the plugin API
// --------------------------------------------------------------------
char const* apiVersionMax() { return VSI_API_VERSION_MAX_STR; }

// --------------------------------------------------------------------
//  Deletes the specified file or directory 
// --------------------------------------------------------------------
char const* name() { return "Vox Scene ExIm"; }

// --------------------------------------------------------------------
//  Deletes the specified file or directory 
// --------------------------------------------------------------------
char const* vendor() { return "Vox"; }

// --------------------------------------------------------------------
//  Returns a description of the plugin
// --------------------------------------------------------------------
char const* description() 
{
    return  "The Vox Scene ExIm plugin provides scene import and export modules "
            "for vox format scene files. These files are XML format specifications of scene data. "
            "See the VoxRender documentation for associated information on required options. "
            ;
}

// --------------------------------------------------------------------
//  Deletes the specified file or directory 
// --------------------------------------------------------------------
void enable() 
{  
    VOX_LOG_INFO(VSI_LOG_CATEGORY, "Enabling the 'Vox Scene ExIm' plugin");
    
    filescope::eximXml  = std::shared_ptr<VoxSceneFile>(new VoxSceneFile(filescope::handle, true ));
    filescope::eximJson = std::shared_ptr<VoxSceneFile>(new VoxSceneFile(filescope::handle, false));

    vox::Scene::registerImportModule(".xml", filescope::eximXml);
    vox::Scene::registerExportModule(".xml", filescope::eximXml);
    vox::Scene::registerImportModule(".json", filescope::eximJson);
    vox::Scene::registerExportModule(".json", filescope::eximJson);
}

// --------------------------------------------------------------------
//  Releases the specified resource module handle
// --------------------------------------------------------------------
void disable() 
{ 
    VOX_LOG_INFO(VSI_LOG_CATEGORY, "Disabling the 'Vox Scene ExIm' plugin");

    vox::Scene::removeImportModule(filescope::eximXml);
    vox::Scene::removeExportModule(filescope::eximXml);
    vox::Scene::removeImportModule(filescope::eximJson);
    vox::Scene::removeExportModule(filescope::eximJson);

    filescope::eximXml.reset();
    filescope::eximJson.reset();
    filescope::handle.reset();
}