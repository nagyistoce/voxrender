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
#include "VoxLib/Scene/Volume.h"

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
    static std::shared_ptr<Volume> execute(Volume & volume, Image3D<float> kernel, 
        Volume::Type type = Volume::Type_End);

    static std::shared_ptr<Volume> execute(Volume & volume, std::vector<float> const& x, std::vector<float> const& y, 
        std::vector<float> const& z, Volume::Type type = Volume::Type_End);

    /** Constructs and returns a gaussian kernel of the given size */
    static std::vector<float> gaussian(float variance, unsigned int size = 0);

    /** Returns the time elapsed during the last convolution operatino */
    static float getElapsedTime();

private:
    Conv();
};

} // namespace volt
} // namespace vox

// End definition
#endif // VOX_VOLT_CONV_H