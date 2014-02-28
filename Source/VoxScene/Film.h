/* ===========================================================================

	Project: Film - render target definition

	Description: Defines the film class used as a render target.

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
#ifndef VOX_FILM_H
#define VOX_FILM_H

// Include Dependencies
#include "VoxLib/Core/CudaCommon.h"
#include "VoxLib/Core/Geometry/Color.h"

// Context change set
#define VOX_CC m_contextChanged = true;

// API namespace
namespace vox
{
    class RenderController;

	/** Film Class */
	class VOX_EXPORT Film
	{
	public:
		Film() : m_height(0), m_width(0) { }

        /** Film height accessor */
        inline size_t height() const { return m_height; }

        /** Film width accessor */
        inline size_t width() const { return m_width; }

        /** Returns the aspect ratio of the film */
        inline float aspectRatio() const { return float(m_width) / m_height; }

        /** Film height modifier */
        inline void setHeight(size_t height) { m_height = height; VOX_CC }

        /** Film width modifier */
        inline void setWidth(size_t width) { m_width = width; VOX_CC }
        
        /** Returns true if the context change flag is set */
        inline bool isDirty() { return m_contextChanged; }

	private:
        friend class RenderController;

		size_t m_height;  ///< Film height
		size_t m_width;   ///< Film width

        bool m_contextChanged; ///< Context change flag
	};
}

#undef VOX_CC

// End definition
#endif // VOX_FILM_H
