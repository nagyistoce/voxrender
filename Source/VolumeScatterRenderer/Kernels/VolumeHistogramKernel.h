/* ===========================================================================

	Project: GPU based Volume Scatter Renderer
    
	Description: Generates histogram information for scene data structures

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
#ifndef VSR_HISTOGRAM_KERNEL_H
#define VSR_HISTOGRAM_KERNEL_H

// Include Headers
#include "VolumeScatterRenderer/Core/Common.h"
#include "VolumeScatterRenderer/Scene/CVolumeBuffer.h"

// VoxLib Dependencies
#include "VoxLib/Core/Geometry.h"

// API namespace
namespace vox
{

/** Volume Histogram Class */
class VolumeHistogramKernel
{
public:
    /** Computes the range of data values present in the volume */
    static Vector2f computeValueRange(std::shared_ptr<Volume> volume);

    /** Generates histogram images for the volume dataset */
    static std::vector<size_t> generateHistogramImages(size_t nBins, std::shared_ptr<Volume> volume);
};

}

// End definition
#endif // VSR_HISTOGRAM_KERNEL_H