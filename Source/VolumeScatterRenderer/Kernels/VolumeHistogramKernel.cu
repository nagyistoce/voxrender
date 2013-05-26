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

// Include Header
#include "VolumeHistogramKernel.h"

// Include Dependencies
#include <limits>

namespace vox {

namespace {
namespace filescope {

    // --------------------------------------------------------------------
    //                        TEXTURE SAMPLERS
    // --------------------------------------------------------------------
    
    texture<UInt8,3,cudaReadModeElementType>  gd_volumeTexE_UInt8;     ///< Volume data texture
    texture<UInt16,3,cudaReadModeElementType> gd_volumeTexE_UInt16;    ///< Volume data texture

    // --------------------------------------------------------------------
    //  Uses vector reduction techniques to compute the max value range
    // --------------------------------------------------------------------
    template<typename T> __global__ void maxValueRangeKernel()
    { 	
	    int x = blockIdx.x * blockDim.x + threadIdx.x;
	    int y = blockIdx.y * blockDim.y + threadIdx.y;
    }
    
    // --------------------------------------------------------------------
    // :TODO:
    // --------------------------------------------------------------------
    template<typename T> Vector2f maxValueRange(size_t elements, UInt8 const* raw)
    {
        Vector<T,2> minMax(std::numeric_limits<T>::max(), static_cast<T>(0));

        T const* data = reinterpret_cast<T const*>(raw);

        for (size_t i = 0; i < elements; i++)
        {
            if (minMax[0] > *data) minMax[0] = *data;
            else if (minMax[1] < *data) minMax[1] = *data;

            data++;
        }

        Vector2f result = static_cast<Vector2f>(minMax) / 
            static_cast<float>(std::numeric_limits<T>::max());

        return result;
    }
    
    // --------------------------------------------------------------------
    // :TODO:
    // --------------------------------------------------------------------
    template<typename T> std::vector<size_t> generateHistogramBins(size_t nBins, size_t elements, UInt8 const* raw)
    {
        Vector2f range = maxValueRange<T>(elements, raw);

        std::vector<size_t> bins(nBins, 0);

        T const* data = reinterpret_cast<T const*>(raw);
        float    max  = static_cast<float>(std::numeric_limits<T>::max());

        for (size_t i = 0; i < elements; i++)
        {
            float  sample           = static_cast<float>(data[i]) / max;
            float  normalizedSample = (sample - range[0]) / (range[1] - range[0]);

            size_t bin = clamp<size_t>(static_cast<size_t>(normalizedSample*nBins), 0, nBins);

            bins[bin]++;
        }

        return bins;
    }

} // namespace filescope
} // namespace anonymous

// ----------------------------------------------------------------------------
//  Computes the minimum and maximum data point values within the volume
// ----------------------------------------------------------------------------
Vector2f VolumeHistogramKernel::computeValueRange(std::shared_ptr<Volume> volume)
{
    size_t elements   = volume->extent().fold<size_t>(1, &mul);
    UInt8 const* data = volume->data();

    switch (volume->type())
    {
        case Volume::Type_UInt8:  return filescope::maxValueRange<UInt8>(elements, data);
        case Volume::Type_UInt16: return filescope::maxValueRange<UInt16>(elements, data);
        default:
            throw Error(__FILE__, __LINE__, VSR_LOG_CATEGORY,
                format("Unsupported volume data type (%1%)", 
                       Volume::typeToString(volume->type())),
                Error_NotImplemented);
    }
}

// ----------------------------------------------------------------------------
//  Generates histogram information for the volume
// ----------------------------------------------------------------------------
std::vector<size_t> VolumeHistogramKernel::generateHistogramImages(size_t nBins, std::shared_ptr<Volume> volume)
{
    size_t elements   = volume->extent().fold<size_t>(1, &mul);
    UInt8 const* data = volume->data();

    switch (volume->type())
    {
        case Volume::Type_UInt8:  return filescope::generateHistogramBins<UInt8>(nBins, elements, data);
        case Volume::Type_UInt16: return filescope::generateHistogramBins<UInt16>(nBins, elements, data);
        default:
            throw Error(__FILE__, __LINE__, VSR_LOG_CATEGORY,
                format("Unsupported volume data type (%1%)", 
                       Volume::typeToString(volume->type())),
                Error_NotImplemented);
    }
}

} // namespace vox