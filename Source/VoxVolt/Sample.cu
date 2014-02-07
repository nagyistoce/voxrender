/* ===========================================================================
                                                                           
   Project: Volume Transform Library                                       
                                                                           
   Description: Performs volume transform operations                       
                                                                           
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
#include "Sample.h"

// Include Dependencies
#include "VoxVolt/Impl/VolumeBlocker.h"
#include "VoxLib/Error/CudaError.h"

// CUDA runtime API header
#include <cuda_runtime_api.h>

#define KERNEL_BLOCK_W		16
#define KERNEL_BLOCK_H		16
#define KERNEL_BLOCK_SIZE   (KERNEL_BLOCK_W * KERNEL_BLOCK_H)

#define MAX_KERNEL_SIZE 5

namespace vox {
namespace volt {

namespace {
namespace filescope {

    __constant__ float gd_kernel[MAX_KERNEL_SIZE*MAX_KERNEL_SIZE*MAX_KERNEL_SIZE];

    static float elapsedTime = 0.0f;

} // namespace filescope
} // namespace anonymous

// ----------------------------------------------------------------------------
//  Executes the convolution kernel on the input volume data
// ----------------------------------------------------------------------------
void Sample::scale(Volume & volume, double shift, double scale)
{   

}

} // namespace volt
} // namespace vox