/* ===========================================================================

	Project: VoxRender - Postprocessing model

	Description: Abstract base class which defines postprocessing modules

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

// Begin definition
#ifndef VOX_POSTPROCESSOR_H
#define VOX_POSTPROCESSOR_H

// VoxLib Dependencies
#include "VoxLib/Core/CudaCommon.h"
#include "VoxLib/Core/Geometry/Color.h"
#include "VoxLib/Core/Logging.h"
#include "VoxLib/Rendering/FrameBuffer.h"
#include "VoxLib/Scene/Scene.h"

// API namespace
namespace vox 
{

/** Abstract Postprocessor Model */
VOX_EXPORT class Postprocessor
{
public:
    virtual ~Postprocessor() { }
};

}

// End definition
#endif // VOX_POSTPROCESSOR_H