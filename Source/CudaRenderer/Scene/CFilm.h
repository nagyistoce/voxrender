/* ===========================================================================

	Project: CUDA Renderer - Device Side Camera

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

// Begin definition
#ifndef CR_CFILM_H
#define CR_CFILM_H

// Common Library Header
#include "CudaRenderer/Core/Common.h"

// VoxLib Dependencies
#include "VoxLib/Core/Geometry.h"
#include "VoxLib/Scene/Scene.h"

// API namespace
namespace vox
{

/** Rendering Film Class */
class CFilm
{
public:
    VOX_HOST_DEVICE CFilm() { }

    /** Constructs a device film for the specified scene */
    VOX_HOST CFilm(Scene const& scene);

    /** Resizes the film to the specified dimensions */

    /** Returns the width of the rendering film in pixels */
    VOX_HOST_DEVICE inline size_t width() const { return m_width; }

    /** Returns the height of the rendering film in pixels */
    VOX_HOST_DEVICE inline size_t height() const { return m_height; }

    /** Returns the stride of the rendering film in bytes */
    VOX_HOST_DEVICE inline size_t stride() const { return m_stride; }

private:
    ColorRgbaHdr * m_framebuffer; ///< Current HDR image buffer
    size_t         m_width;       ///< Width of the framebuffer
    size_t         m_height;      ///< Height of the framebuffer
    size_t         m_stride;      ///< Stride of the framebuffer
};

}

// End definition
#endif // CR_CFILM_H