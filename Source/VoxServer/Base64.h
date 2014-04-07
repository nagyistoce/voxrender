/* ===========================================================================

	Project: VoxServer

	Description: Implements a WebSocket based server for interactive rendering

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
#ifndef VOX_BASE64_H
#define VOX_BASE64_H

// Include Dependencies
#include "VoxServer/Common.h"
#include "VoxLib/Core/Types.h"

namespace vox {

/** Implements Base64 encode and decode functionality */
class Base64
{
public:
    /** Encodes a string using Base64 encoding */
    static String encode(String const& data);
    
    /** Decodes a string in a Base64 encoding */
    static String decode(String const& data);
};

} // namespace vox

#endif // VOX_BASE64_H
