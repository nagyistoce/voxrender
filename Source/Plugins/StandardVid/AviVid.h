/* ===========================================================================

    Project: Standard Video Import/Export
    
	Description: Provides an import/export module for video formats

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

// Begin definition
#ifndef VOX_SVID_AVI_VID_H
#define VOX_SVID_AVI_VID_H

// Include Dependencies
#include "StandardVid/Common.h"
#include "VoxLib/Core/Format.h"
#include "VoxLib/Video/VidCodec.h"

// API namespace
namespace vox 
{

/**
 * Standard video file import / export module
 *
 * This module is compatible with the abstract video codec interface
 */
class VOX_SVID_EXPORT AviVid : public VideoCodec
{
public:
    AviVid(std::shared_ptr<void> handle) : m_handle(handle) { }
    
private:
    std::shared_ptr<void> m_handle; ///< Plugin handle to track this DLL's usage
};

}

// End definition
#endif // VOX_SVID_AVI_VID_H