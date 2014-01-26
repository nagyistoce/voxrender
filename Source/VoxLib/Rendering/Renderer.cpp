/* ===========================================================================

	Project: VoxRender - Rendering model

	Description: Abstract base classes which define rendering modules

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
#include "Renderer.h"

// Include Dependencies
#include "VoxLib/Core/Logging.h"

// API namespace
namespace vox
{

// --------------------------------------------------------------------
//  Rebuilds the device geometry structures 
// --------------------------------------------------------------------
bool Renderer::exception(std::exception_ptr & exception) 
{ 
    try
    {
        std::rethrow_exception(exception);
    }

    catch (Error const& error)
    {
        Logger::addEntry(error);
    }

    catch (std::exception const& error)
    {
        Logger::addEntry(Severity_Error, Error_Unknown, "VOX",
                            error.what(), __FILE__, __LINE__);
    }

    catch (...) 
    { 
        Logger::addEntry(Severity_Error, Error_Unknown, "VOX",
                            "Unknown Render Thread Failure",
                            __FILE__, __LINE__);
    }

    return false;
}

}