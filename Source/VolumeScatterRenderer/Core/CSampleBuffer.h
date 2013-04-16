/* ===========================================================================

	Project: VoxRender - Device Sample Data Buffer

	Description: Defines the sample buffer management class

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
#ifndef VSR_CSAMPLE_BUFFER_H
#define VSR_CSAMPLE_BUFFER_H

// Common Library Header
#include "VolumeScatterRenderer/Core/Common.h"

// VoxLib Dependencies
#include "VoxLib/Core/Geometry.h"

// Device representations of scene components
#include "VolumeScatterRenderer/Core/CBuffer.h"

// API namespace
namespace vox
{

/** CUDA Sample Buffer Class (Device Base) */
class CSampleBuffer2D
{
public:
    /** Initializes the sample buffer members */
    VOX_HOST void init() 
    {
        m_samplingBuffer.init();
        m_varianceBuffer.init();
    }

    /** Resets the sample buffers */
    VOX_HOST void reset()
    {
        m_samplingBuffer.reset();
        m_varianceBuffer.reset();
    }

    /** 
     * Resizes the sample buffer 
     * 
     * If the clear flag is not set, then the internal buffers
     * will be cleared. Otherwise, the buffer will be resampled
     * to accomodate the changes in size.
     */
    VOX_HOST void resize(size_t width, size_t height, bool clear = true)
    {
        m_samplingBuffer.resize(width, height);
        m_varianceBuffer.resize(width, height);
        m_samplingBuffer.clear();
        m_varianceBuffer.clear();
    }

    /** Clears the image buffer */
    VOX_HOST void clear()
    {
        m_samplingBuffer.clear();
        m_varianceBuffer.clear();
    }

    /** Returns the sample buffer's mean+density data buffer */
    VOX_HOST CBuffer2D<ColorLabxHdr> & meanBuffer() { return m_samplingBuffer; }

    /** Returns the sample buffer's variance data buffer */
    VOX_HOST CBuffer2D<float> & varianceBuffer() { return m_varianceBuffer; }

    /** Returns the width of the sample buffers internal image buffers */
    VOX_HOST_DEVICE size_t width() const { return m_samplingBuffer.width(); }

    /** Returns the height of the sample buffers internal image buffers */
    VOX_HOST_DEVICE size_t height() const { return m_samplingBuffer.height(); }

    /** Pushes a new sample into the buffer at the specified image coordinates */
    VOX_HOST_DEVICE void push(size_t x, size_t y, ColorLabxHdr const& sample)
    {
        // :TODO:
        // Atomically inc the sample density at the location
        float samples = m_samplingBuffer.at(x,y).x + 1.0f;
        m_samplingBuffer.at(x,y).x = samples;

        if (samples == 1.0f)
        {
            // Retrieve the current mean sample 
            ColorLabxHdr & mean = m_samplingBuffer.at(x,y);

            // Set the mean to the initial sample value
            mean.l = sample.l; 
            mean.a = sample.a; 
            mean.b = sample.b;
        }
        else
        {
            // Retrieve the current mean sample 
            ColorLabxHdr & mean = m_samplingBuffer.at(x,y);
            
            // Compute the distance between the old mean and X
            float distOldMean = ColorLabxHdr::distance(sample, mean);

            // Update the running average color sample
            mean.l = runningMean(samples, mean.l, sample.l);
            mean.a = runningMean(samples, mean.a, sample.a);
            mean.b = runningMean(samples, mean.b, sample.b);
            
            // Compute the distance between the new mean and X
            float distNewMean = ColorLabxHdr::distance(sample, mean);

            // Update the running variance factor
            m_varianceBuffer.at(x,y) += distNewMean*distOldMean;
        }
    }

    /** Returns the mean RGBX sample value */
    VOX_HOST_DEVICE ColorLabxHdr const& at(size_t x, size_t y) const
    {
        return m_samplingBuffer.at(x, y);
    }

    /** Returns the sample density of the specified sample */
    VOX_HOST_DEVICE float sampleDensity(size_t x, size_t y) const
    {
        return m_samplingBuffer.at(x,y).x;
    }

    /** Returns the variance at a buffer location */
    VOX_HOST_DEVICE float variance(size_t x, size_t y) const
    {
        float n = sampleDensity(x,y);

        return (n > 0.0f) ? m_varianceBuffer.at(x,y) / (n-1.f) : 0.f;
    }

    /** Returns the standard deviation */
    VOX_HOST_DEVICE float standardDeviation(size_t x, size_t y) const
    {
        return sqrtf(variance(x,y));
    }

private:
    CBuffer2D<ColorLabxHdr> m_samplingBuffer;  ///< RGB sample mean and X = sample density
    CBuffer2D<float>        m_varianceBuffer;  ///< Variance tracking buffer, stores running sum(dist^2)

    /** Computes the running average of a running base average and a value */
    VOX_HOST_DEVICE float runningMean(float samples, float base, float value)
    {
        return base + (value - base) / samples;
    }
};

}

// End definition
#endif // VSR_CSAMPLE_BUFFER_H