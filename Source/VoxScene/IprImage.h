/* ===========================================================================

	Project: VoxRender

	Description: Contains information on a partial render 

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
#ifndef VOX_IPR_IMAGE_H
#define VOX_IPR_IMAGE_H

// Internal Dependencies
#include "VoxScene/Common.h"
#include "VoxLib/Core/Geometry/Vector.h"
#include "VoxLib/Core/Geometry/Image.h"
#include "VoxLib/Core/Geometry/Color.h"

// API namespace
namespace vox
{
	/** Rendering frame buffer class */
    class VOXS_EXPORT IprImage
	{
	public:
        Image<ColorLabxHdr> sampleBuffer;
        Image<float> statsBuffer;

    public:
        void write(std::ostream & in);
        void read(std::istream & out);
	};
}

// End definition
#endif // VOX_IPR_IMAGE_H
