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
#ifndef VOX_VOLT_HISTOGRAM_VOLUME_H
#define VOX_VOLT_HISTOGRAM_VOLUME_H

// Include Dependencies
#include "VoxVolt/Common.h"
#include "VoxScene/Volume.h"
#include "VoxLib/IO/OptionSet.h"

// API namespace
namespace vox {
namespace volt {

/** Manages a meta-volume for a volume dataset */
class VOX_VOLT_EXPORT HistogramVolume
{
public:
    /**
     * Generates a histogram volume from an input volume dataset
     *
     * @param volume  The input volume data set
     * @param size    The dimensions of the histogram volume
     * @param options [in out] allows configuration of the histogram and stats output
     * @return A handle to the histogram volume
     */
    static std::shared_ptr<Volume> build(
        std::shared_ptr<Volume> volume, 
        Vector3s const& size  = Vector3s(256, 256, 256),
        OptionSet const& options = OptionSet()
        );

private:
    HistogramVolume();
};

} // namespace volt
} // namespace vox

// End definition
#endif // VOX_VOLT_HISTOGRAM_VOLUME_H