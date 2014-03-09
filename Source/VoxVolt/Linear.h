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
#ifndef VOX_VOLT_LINEAR_H
#define VOX_VOLT_LINEAR_H

// Include Dependencies
#include "VoxVolt/Common.h"
#include "VoxLib/Core/Geometry/Image3D.h"
#include "VoxScene/Volume.h"

// API namespace
namespace vox {
namespace volt {

/** Implements transforms for convolution operations */
class VOX_VOLT_EXPORT Linear
{
public:
    /**
     * Performs a linear transformation operation on the volume data set (in place)
     *
     * @param volume The input volume data set
     * @param shift  The amount to shift each value by
     * @param scale  A scale factor for each data value
     */
    static std::shared_ptr<Volume> execute(std::shared_ptr<Volume> volume, double shift, double scale);

    /**
     * Performs a linear transformation operation on the volume data set
     *
     * @param volume The input volume data set
     * @param shift  The amount to shift each value by
     * @param scale  A scale factor for each data value
     * @param type   The target type of the output volume data set
     */
    static std::shared_ptr<Volume> execute(
        std::shared_ptr<Volume> volume, 
        double shift, 
        double scale, 
        Volume::Type type);

private:
    Linear();
};

} // namespace volt
} // namespace vox

// End definition
#endif // VOX_VOLT_LINEAR_H