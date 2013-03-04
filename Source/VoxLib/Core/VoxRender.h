/* ===========================================================================

	Project: VoxRender - GPU Based Real-Time Volume Rendering Library

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
#ifndef VOX_VOXRENDER_H
#define VOX_VOXRENDER_H

// Vox Common Header
#include "VoxLib/Core/Common.h"

// Core Includes
#include "VoxLib/Core/Format.h"
#include "VoxLib/Core/Logging.h"
#include "VoxLib/Core/Devices.h"
#include "VoxLib/Core/System.h"
#include "VoxLib/Core/Types.h"

// Geometry Includes
#include "VoxLib/Core/Geometry.h"

// IO Library Includes
#include "VoxLib/IO/Resource.h"
#include "VoxLib/IO/ResourceId.h"

// Rendering Includes
#include "VoxLib/Rendering/FrameBuffer.h"
#include "VoxLib/Rendering/RenderController.h"
#include "VoxLib/Rendering/Renderer.h"

// Scene Includes
#include "VoxLib/Scene/Scene.h"
#include "VoxLib/Scene/Camera.h"
#include "VoxLib/Scene/Volume.h"
#include "VoxLib/Scene/Film.h"
#include "VoxLib/Scene/Light.h"
#include "VoxLib/Scene/Transfer.h"

// End definition
#endif // VOX_VOXRENDER_H