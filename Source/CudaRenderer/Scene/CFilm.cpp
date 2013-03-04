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
#include "CFilm.h"

// Include Dependencies
#include "VoxLib/Scene/Film.h"

// API namespace
namespace vox
{

// --------------------------------------------------------------------
//  Allocates a device side framebuffer for rendering operations
// --------------------------------------------------------------------
CFilm::CFilm(Scene const& scene) :
    m_width(scene.film->width()), 
    m_height(scene.film->height())
{
    // Allocate the device render data buffer
    cudaMallocPitch((void**)&m_framebuffer, 
                    &m_stride, 
                    m_width, 
                    m_height);
}

} // namespace vox