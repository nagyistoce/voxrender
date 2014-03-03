/* ===========================================================================

	Project: VoxRender - Functors

	Defines some basic functors for simplifying container operations such as
    folding and mapping.

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
#include "Functors.h"

namespace vox {

// --------------------------------------------------------------------
//  Implements a convenience array deleter for std::shared_ptr
// --------------------------------------------------------------------
std::shared_ptr<UInt8> makeSharedArray(size_t bytes)
{
    return std::shared_ptr<UInt8>(new UInt8[bytes], &arrayDeleter);
}

// --------------------------------------------------------------------
//  Implements a convenience array deleter for std::shared_ptr
// --------------------------------------------------------------------
void arrayDeleter(void* data) { delete[] data; }

// --------------------------------------------------------------------
//  Implements a convenience null deleter for std::shared_ptr
// --------------------------------------------------------------------
void nullDeleter(void* data) { }

} // namespace vox