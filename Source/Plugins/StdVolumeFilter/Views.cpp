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
#include "Views.h"

// API dependencies
#include "VoxLib/Core/Common.h"
#include "VoxLib/Core/Format.h"
#include "VoxScene/Camera.h"

namespace vox {

using namespace volt;

const char* View::dirStrings[] = { "Front", "Left", "Top", "Back", "Right", "Bottom" };

namespace {
namespace filescope {

} // namespace filescope
} // namespace anonymous

// ----------------------------------------------------------------------------
//  Auto-aligns the camera to the specified viewing plane
// ----------------------------------------------------------------------------
void View::execute(Scene & scene, OptionSet const& params)
{
    SceneLock lock(scene.camera); // Lock the scene for editing

    static const double PAD_FACTOR = 1.2;

    auto extent  = scene.volume->extent();
    auto spacing = scene.volume->spacing();
    auto aspect  = scene.camera->aspectRatio();

    double size;
    switch (m_dir)
    {
    case Dir_Front: case Dir_Back:
        size = high(extent[0]*spacing[0], extent[1]*spacing[1]/aspect);
        break; 
    case Dir_Top: case Dir_Bottom:
        size = high(extent[0]*spacing[0], extent[2]*spacing[2]/aspect);
        break;
    case Dir_Left: case Dir_Right:
        size = high(extent[2]*spacing[2], extent[1]*spacing[1]/aspect); 
        break;
    default:
        throw Error(__FILE__, __LINE__, SVF_LOG_CATEGORY, "Invalid viewing plane", Error_Bug);
    }

    auto distance = 0.5 * (size * PAD_FACTOR) / tan(scene.camera->fieldOfView() / 2.0);

    auto half = Vector4f(extent) * spacing * 0.5;
    switch (m_dir)
    {
    case Dir_Front:
        scene.camera->setPosition(Vector3f(0, 0, half[2]+distance));
        scene.camera->setEye(Vector3f(0, 0, -1));
        scene.camera->setUp(Vector3f(0, 1, 0));
        scene.camera->setRight(Vector3f(1, 0, 0));
        break;
    case Dir_Back:
        scene.camera->setPosition(Vector3f(0, 0, -half[2]-distance));
        scene.camera->setEye(Vector3f(0, 0, 1));
        scene.camera->setUp(Vector3f(0, 1, 0));
        scene.camera->setRight(Vector3f(-1, 0, 0));
        break;
    case Dir_Top:
        scene.camera->setPosition(Vector3f(0, half[1]+distance, 0));
        scene.camera->setEye(Vector3f(0, -1, 0));
        scene.camera->setUp(Vector3f(0, 0, -1));
        scene.camera->setRight(Vector3f(1, 0, 0));
        break;
    case Dir_Bottom:
        scene.camera->setPosition(Vector3f(0, -half[1]-distance, 0));
        scene.camera->setEye(Vector3f(0, 1, 0));
        scene.camera->setUp(Vector3f(0, 0, -1));
        scene.camera->setRight(Vector3f(-1, 0, 0));
        break;
    case Dir_Left:
        scene.camera->setPosition(Vector3f(-half[0]-distance, 0, 0));
        scene.camera->setEye(Vector3f(1, 0, 0));
        scene.camera->setUp(Vector3f(0, 1, 0));
        scene.camera->setRight(Vector3f(0, 0, 1));
        break;
    case Dir_Right:
        scene.camera->setPosition(Vector3f(half[0]+distance, 0, 0));
        scene.camera->setEye(Vector3f(-1, 0, 0));
        scene.camera->setUp(Vector3f(0, 1, 0));
        scene.camera->setRight(Vector3f(0, 0, -1));
        break;
    default:
        throw Error(__FILE__, __LINE__, SVF_LOG_CATEGORY, "Invalid viewing plane", Error_Bug);
    }

    scene.camera->setPosition(scene.camera->position() + scene.volume->offset());
}

} // namespace vox