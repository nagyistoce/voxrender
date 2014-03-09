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
#include "Linear.h"

// Include Dependencies
#include "VoxLib/Core/Logging.h"

// Std library
#include <chrono>

namespace vox {
namespace volt {

    namespace {
    namespace filescope {
    
        // ----------------------------------------------------------------------------
        //  Performs a type specific scale operation
        // ----------------------------------------------------------------------------
        template<typename T> 
        void doTransformIP(T * data, size_t voxels, double shift, double scale)
        {
            double maxVal = std::numeric_limits<T>::max();
            double minVal = std::numeric_limits<T>::min();

            for (size_t i = 0; i < voxels; ++i)
            {
                double val = (static_cast<double>(*data) + shift) * scale; 
                *data = static_cast<T>(clamp(val, minVal, maxVal));
                ++data;
            }
        }
        
        // ----------------------------------------------------------------------------
        //  Performs a type specific scale operation
        // ----------------------------------------------------------------------------
        template<typename T, typename S> 
        void doTransformNIP(T * data, S * dataO, size_t voxels, double shift, double scale)
        {
            double maxVal = std::numeric_limits<S>::max();
            double minVal = std::numeric_limits<S>::min();

            for (size_t i = 0; i < voxels; ++i)
            {
                auto val = (static_cast<double>(*data) + shift) * scale; 
                *dataO = static_cast<S>(clamp(val, minVal, maxVal));
                ++data;
                ++dataO;
            }
        }

        // ----------------------------------------------------------------------------
        //  Performs the type specific scale operation between volumes
        // ----------------------------------------------------------------------------
        template<typename T> 
        void doTransform(T * data, UInt8 * dataO, size_t voxels, double shift, double scale, Volume::Type type)
        {
            switch (type)
            {
            case Volume::Type_Int8:    filescope::doTransformNIP(data, (Int8*)   dataO, voxels, shift, scale); break;
            case Volume::Type_UInt8:   filescope::doTransformNIP(data, (UInt8*)  dataO, voxels, shift, scale); break;
            case Volume::Type_UInt16:  filescope::doTransformNIP(data, (UInt16*) dataO, voxels, shift, scale); break;
            case Volume::Type_Int16:   filescope::doTransformNIP(data, (Int16*)  dataO, voxels, shift, scale); break;
            case Volume::Type_UInt32:  filescope::doTransformNIP(data, (UInt32*) dataO, voxels, shift, scale); break;
            case Volume::Type_Int32:   filescope::doTransformNIP(data, (Int32*)  dataO, voxels, shift, scale); break;
            case Volume::Type_Float32: filescope::doTransformNIP(data, (Float32*)dataO, voxels, shift, scale); break;
            default: throw Error(__FILE__, __LINE__, VOLT_LOG_CAT, "Unsupported volume data type", Error_NotImplemented);
            }
        }
    }
    }

// ----------------------------------------------------------------------------
//  Performs a simple linear transformation of the data (always on the CPU)
// ----------------------------------------------------------------------------
std::shared_ptr<Volume> Linear::execute(std::shared_ptr<Volume> volume, double shift, double scale)
{
    // Begin tracking the performance time
    auto tbeg = std::chrono::high_resolution_clock::now();
    
    // Perform the shift/scale transformation
    auto voxels = volume->extent().fold(mul);
    auto data   = volume->mutableData();
    switch (volume->type())
    {
    case Volume::Type_Int8:    filescope::doTransformIP((Int8*)   data, voxels, shift, scale); break;
    case Volume::Type_UInt8:   filescope::doTransformIP((UInt8*)  data, voxels, shift, scale); break;
    case Volume::Type_UInt16:  filescope::doTransformIP((UInt16*) data, voxels, shift, scale); break;
    case Volume::Type_Int16:   filescope::doTransformIP((Int16*)  data, voxels, shift, scale); break;
    case Volume::Type_UInt32:  filescope::doTransformIP((UInt32*) data, voxels, shift, scale); break;
    case Volume::Type_Int32:   filescope::doTransformIP((Int32*)  data, voxels, shift, scale); break;
    case Volume::Type_Float32: filescope::doTransformIP((Float32*)data, voxels, shift, scale); break;
    default: throw Error(__FILE__, __LINE__, VOLT_LOG_CAT, "Unsupported volume data type", Error_NotImplemented);
    }

    // Compute the time elapsed during execution
    auto tend = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(tend-tbeg);
    VOX_LOG_INFO(VOLT_LOG_CAT, format("Linear transformation completed in %1% ms", time.count()));

    return volume;
}


// ----------------------------------------------------------------------------
//  Performs a simple linear transformation of the data (always on the CPU)
// ----------------------------------------------------------------------------
std::shared_ptr<Volume> Linear::execute(std::shared_ptr<Volume> volume, double shift, double scale, Volume::Type type)
{
    // Begin tracking the performance time
    auto tbeg = std::chrono::high_resolution_clock::now();
    
    // Perform the shift/scale transformation
    auto voxels = volume->extent().fold(mul);
    auto data   = volume->mutableData();
    auto dataO  = makeSharedArray(voxels * Volume::typeToSize(type));
    switch (volume->type())
    {
    case Volume::Type_Int8:    filescope::doTransform((Int8*)   data, dataO.get(), voxels, shift, scale, type); break;
    case Volume::Type_UInt8:   filescope::doTransform((UInt8*)  data, dataO.get(), voxels, shift, scale, type); break;
    case Volume::Type_UInt16:  filescope::doTransform((UInt16*) data, dataO.get(), voxels, shift, scale, type); break;
    case Volume::Type_Int16:   filescope::doTransform((Int16*)  data, dataO.get(), voxels, shift, scale, type); break;
    case Volume::Type_UInt32:  filescope::doTransform((UInt32*) data, dataO.get(), voxels, shift, scale, type); break;
    case Volume::Type_Int32:   filescope::doTransform((Int32*)  data, dataO.get(), voxels, shift, scale, type); break;
    case Volume::Type_Float32: filescope::doTransform((Float32*)data, dataO.get(), voxels, shift, scale, type); break;
    default: throw Error(__FILE__, __LINE__, VOLT_LOG_CAT, "Unsupported volume data type", Error_NotImplemented);
    }

    // Compute the time elapsed during execution
    auto tend = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(tend-tbeg);
    VOX_LOG_INFO(VOLT_LOG_CAT, format("Linear transformation completed in %1% ms", time.count()));

    return Volume::create(dataO, volume->extent(), volume->spacing(), volume->offset(), type);
}

} // namespace volt
} // namespace vox
