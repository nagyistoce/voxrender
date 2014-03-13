/* ===========================================================================

	Project: Standard Volume Filters
    
	Description: Exposes some standard filters provided by the volt library

    Copyright (C) 2014 Lucas Sherman

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
#include "StdVolumeFilter/Common.h"
#include "StdVolumeFilter/Filters.h"
#include "VoxLib/Core/Logging.h"
#include "VoxLib/Plugin/PluginManager.h"
#include "VoxVolt/Filter.h"

using namespace vox;

namespace {
namespace filescope {

    static std::shared_ptr<volt::Filter> gauss;
    static std::shared_ptr<volt::Filter> mean;
    static std::shared_ptr<volt::Filter> laplace;
    static std::shared_ptr<volt::Filter> lanczos;
    static std::shared_ptr<volt::Filter> linear;
    static std::shared_ptr<volt::Filter> crop;
    std::shared_ptr<void> handle;

} // namespace filescope
} // namespace anonymous

// --------------------------------------------------------------------
//  Deletes the specified file or directory 
// --------------------------------------------------------------------
void initPlugin() 
{
    VOX_LOG_INFO(SVF_LOG_CATEGORY, "Loading the 'Vox.Std Volume Filter' plugin"); 
    
    filescope::handle = PluginManager::instance().acquirePluginHandle();
}

// --------------------------------------------------------------------
//  Deletes the specified file or directory 
// --------------------------------------------------------------------
void freePlugin()
{ 
    VOX_LOG_INFO(SVF_LOG_CATEGORY, "Unloading the 'Vox.Std Volume Filter' plugin");
}

// --------------------------------------------------------------------
//  Returns the dot delimited version string for this build
// --------------------------------------------------------------------
char const* version() { return SVF_VERSION_STRING; }

// --------------------------------------------------------------------
//  Returns a reference URL for the plugin
// --------------------------------------------------------------------
char const* referenceUrl() { return "http://code.google.com/p/voxrender/"; }

// --------------------------------------------------------------------
//  Returns the minimum compatible version of the plugin API
// --------------------------------------------------------------------
char const* apiVersionMin() { return SVF_API_VERSION_MIN_STR; }

// --------------------------------------------------------------------
//  Returns the minimum compatible version of the plugin API
// --------------------------------------------------------------------
char const* apiVersionMax() { return SVF_API_VERSION_MAX_STR; }

// --------------------------------------------------------------------
//  Deletes the specified file or directory 
// --------------------------------------------------------------------
char const* name() { return "Std Volume Filter"; }

// --------------------------------------------------------------------
//  Deletes the specified file or directory 
// --------------------------------------------------------------------
char const* vendor() { return "Vox"; }

// --------------------------------------------------------------------
//  Returns a description of the plugin
// --------------------------------------------------------------------
char const* description() 
{
    return  "The Standard Volume Filter plugin provides simple filtering options for volume data."
            ;
}

// --------------------------------------------------------------------
//  Deletes the specified file or directory 
// --------------------------------------------------------------------
void enable() 
{  
    VOX_LOG_INFO(SVF_LOG_CATEGORY, "Enabling the 'Vox.Std Volume Filter' plugin");

    filescope::gauss   = std::shared_ptr<volt::Filter>(new GaussFilter(filescope::handle));
    filescope::mean    = std::shared_ptr<volt::Filter>(new MeanFilter(filescope::handle));
    filescope::laplace = std::shared_ptr<volt::Filter>(new LaplaceFilter(filescope::handle));
    filescope::lanczos = std::shared_ptr<volt::Filter>(new LanczosFilter(filescope::handle));
    filescope::linear  = std::shared_ptr<volt::Filter>(new LinearFilter(filescope::handle));
    filescope::crop    = std::shared_ptr<volt::Filter>(new CropFilter(filescope::handle));

    volt::FilterManager::instance().add(filescope::gauss);
    volt::FilterManager::instance().add(filescope::mean);
    volt::FilterManager::instance().add(filescope::laplace);
    volt::FilterManager::instance().add(filescope::lanczos);
    volt::FilterManager::instance().add(filescope::linear);
    volt::FilterManager::instance().add(filescope::crop);
}

// --------------------------------------------------------------------
//  Releases the specified resource module handle
// --------------------------------------------------------------------
void disable() 
{ 
    VOX_LOG_INFO(SVF_LOG_CATEGORY, "Disabling the 'Vox.Std Volume Filter' plugin");

    volt::FilterManager::instance().remove(filescope::gauss);
    volt::FilterManager::instance().remove(filescope::mean);
    volt::FilterManager::instance().remove(filescope::laplace);
    volt::FilterManager::instance().remove(filescope::lanczos);
    volt::FilterManager::instance().remove(filescope::linear);
    volt::FilterManager::instance().remove(filescope::crop);
    
    filescope::crop.reset();
    filescope::gauss.reset();
    filescope::mean.reset();
    filescope::laplace.reset();
    filescope::lanczos.reset();
    filescope::linear.reset();
    filescope::handle.reset();
}