/* ===========================================================================

	Project: GPU based Volume Scatter Renderer
    
	Description: Performs tonemapping on a HDR input buffer

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
#ifndef VSR_TONEMAP_KERNEL_H
#define VSR_TONEMAP_KERNEL_H

// Common Library Header
#include "VolumeScatterRenderer/Core/Common.h"

// API namespace
namespace vox 
{
    
// Forward Declarations
template<typename T> class CImgBuffer2D;
class CSampleBuffer2D;
struct ColorRgbaLdr;

/** Defines the interface for building and executing render kernels */
class TonemapKernel
{
public:
    /** Executes the tonemapping kernel on the device */
    static void execute(CSampleBuffer2D sampleBuffer, CImgBuffer2D<ColorRgbaLdr> imageBuffer);

    /** Returns the time for the last kernel execution */
    static float getTime() { return m_elapsedTime; }

private:
    static float m_elapsedTime; ///< Kernel execution time
};

}

// End definition
#endif // VSR_TONEMAP_KERNEL_H