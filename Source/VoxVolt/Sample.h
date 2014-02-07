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
#ifndef VOX_VOLT_SAMPLE_H
#define VOX_VOLT_SAMPLE_H

// Include Dependencies
#include "VoxVolt/Common.h"
#include "VoxLib/Core/Geometry/Image3D.h"
#include "VoxLib/Scene/Volume.h"

// API namespace
namespace vox {

    class Volume;
    
namespace volt {

/** Implements transforms for convolution operations */
class VOX_VOLT_EXPORT Sample
{
public:
    /**
     * Resamples the volume by downsampling or upsampling.
     *
     * The resampling will perform low pass filtering in the 3 spatial dimensions
     * but will not resample or interpolate the 4th dimension. (Typically time)
     *
     * @param volume  The input volume data set
     * @param newSize The extent of the output volume
     */
    static std::shared_ptr<Volume> resize(Volume const& volume, Vector4u newSize);

    /** 
     * Changes the underlying format of the volume to the type specified. This is equivalent to
     * performing a scale operation to the output type.
     *
     * @param volume 
     */
    static std::shared_ptr<Volume> changeType(Volume const& volume, Volume::Type outType = Volume::Type_Begin);
    
    /** Performs a linear transformation of the volume data in place */
    static void scale(Volume & volume, double shift, double scale);

    /** Performs a linear transformation of the volume data and retypes */
    static std::shared_ptr<Volume> scale(Volume const& volume, double shift, double scale, Volume::Type typeOut);

private:
    Sample();
};

} // namespace volt
} // namespace vox

// End definition
#endif // VOX_VOLT_SAMPLE_H