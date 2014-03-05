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

// Begin definition
#ifndef VOX_VOLT_CONV_H
#define VOX_VOLT_CONV_H

// Include Dependencies
#include "VoxVolt/Common.h"
#include "VoxLib/Core/Geometry/Image3D.h"
#include "VoxScene/Volume.h"

// API namespace
namespace vox {

    class Volume;
    
namespace volt {

/** Implements transforms for convolution operations */
class VOX_VOLT_EXPORT Conv
{
public:
    /**
     * Performs a convolution operation on the volume data set
     *
     * @param volume The input volume data set
     * @param kernel The volume convoltution kernel
     * @param type   The target type of the output volume data set
     */
    static std::shared_ptr<Volume> execute(
        std::shared_ptr<Volume> volume, 
        Image3D<float> kernel, 
        Volume::Type type = Volume::Type_End);

    /** Performs a convolution operation on the volume data set */
    static std::shared_ptr<Volume> execute(
        std::shared_ptr<Volume> volume, 
        std::vector<float> const& x, 
        std::vector<float> const& y, 
        std::vector<float> const& z, 
        Volume::Type type = Volume::Type_End);

    /** 
     * Constructs and returns a gaussian kernel of the given size (seperable)
     *
     * @param out      [out] The gaussian kernel vector 
     * @param variance The variance of the gaussian
     * @param size     The size of the output, or 0 if the size should be fit to the variance
     */
    static void makeGaussianKernel(std::vector<float> & out, float variance, unsigned int size = 0);

    /** Constructs and returns a windows sinc filter of the given size (seperable)
     *
     * @param out      [out] The gaussian kernel vector 
     * @param freq     The scaling factor in the sinc function     
     * @param size     The size of the output, or 0 if the size should be fit to 5 lobes
     */
    static void makeSincKernel(std::vector<float> & out, float freq, unsigned int size = 0);

    /** Constructs and returns a mean filter kernel of the given size (seperable) */
    static void makeMeanKernel(std::vector<float> & out, unsigned int size);

    /** Constructs a laplacian kernel */
    static void makeLaplaceKernel(Image3D<float> & kernel);

private:
    Conv();
};

} // namespace volt
} // namespace vox

// End definition
#endif // VOX_VOLT_CONV_H