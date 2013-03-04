/* ===========================================================================

	Project: VoxRender - Device Side Camera

	Description: Defines a 3D Camera for use in rendering

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
#include "CCamera.h"

// Include Dependencies
#include "VoxLib/Scene/Camera.h"
#include "VoxLib/Scene/Film.h"

// API namespace
namespace vox
{

// --------------------------------------------------------------------
//  Constructs a camera object for device side rendering 
// --------------------------------------------------------------------
CCamera::CCamera(Scene const& scene) :
    m_focalDistance (scene.camera->focalDistance()),
    m_apertureSize  (scene.camera->apertureSize()),
    m_pos           (scene.camera->position()),
    m_eye           (scene.camera->eye()),
    m_right         (scene.camera->right()),
    m_up            (scene.camera->up())
{
    size_t width  = scene.film->width();
    size_t height = scene.film->height();

    // Precompute screen sampling parameters
    float wo = tanf(scene.camera->fieldOfView() / 2.0f);
    float ho = wo * height / width;
    m_screenUpperLeft[0] = -wo; 
    m_screenUpperLeft[1] = -ho;
    m_screenPerPixel[0]  = 2.0f * wo / width;
    m_screenPerPixel[1]  = 2.0f * ho / height;
}

}