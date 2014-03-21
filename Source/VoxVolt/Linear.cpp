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
std::shared_ptr<Volume> Linear::shiftScale(std::shared_ptr<Volume> volume, double shift, double scale)
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
    VOX_LOG_INFO(VOLT_LOG_CAT, format("Shift+Scale transformation completed in %1% ms", time.count()));

    return volume;
}


// ----------------------------------------------------------------------------
//  Performs a simple linear transformation of the data (always on the CPU)
// ----------------------------------------------------------------------------
std::shared_ptr<Volume> Linear::shiftScale(std::shared_ptr<Volume> volume, double shift, double scale, Volume::Type type)
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
    VOX_LOG_INFO(VOLT_LOG_CAT, format("Shift + Scale transformation completed in %1% ms", time.count()));

    return Volume::create(dataO, volume->extent(), volume->spacing(), volume->offset(), type);
}

// ----------------------------------------------------------------------------
//  Crops the volume to the specified extent
// ----------------------------------------------------------------------------
std::shared_ptr<Volume> Linear::crop(std::shared_ptr<Volume> volume, 
    Vector4 const& newOrigin, Vector4s newExtent, void const* value)
{
    // Begin tracking the performance time
    auto tbeg = std::chrono::high_resolution_clock::now();
    
    // Setup the write volume data buffer
    auto extent   = volume->extent();
    auto voxels   = newExtent.fold(&mul);
    auto voxSize  = Volume::typeToSize(volume->type());
    
    std::shared_ptr<UInt8> data = makeSharedArray(voxels*voxSize);
    memset(data.get(), 0, voxels*voxSize);

    // Determine the read/write volume parameters
    auto readPtr  = volume->mutableData();
    auto writePtr = data.get();
    
    Vector3s rStride;
    rStride[0] = extent[0] * voxSize;
    rStride[1] = extent[1] * rStride[0];
    rStride[2] = extent[2] * rStride[1];

    Vector3s wStride;
    wStride[0] = newExtent[0] * voxSize;
    wStride[1] = newExtent[1] * wStride[0];
    wStride[2] = newExtent[2] * wStride[1];

    // Determine the ranges of the base volume dimensions which are copied 
    Vector4 copyOffset(
        high(newOrigin[0], 0),
        high(newOrigin[1], 0),
        high(newOrigin[2], 0),
        high(newOrigin[3], 0));

    Vector4 copyRange(
        low<int>(newOrigin[0] + newExtent[0], extent[0]) - copyOffset[0],
        low<int>(newOrigin[1] + newExtent[1], extent[1]) - copyOffset[1],
        low<int>(newOrigin[2] + newExtent[2], extent[2]) - copyOffset[2],
        low<int>(newOrigin[3] + newExtent[3], extent[3]) - copyOffset[3]);

    // Copy all rows within the uncropped portion of the base volume
    size_t copyLength  = copyRange[0] * voxSize;
    size_t readOffset  = (newOrigin[0] > 0) ?   newOrigin[0]*voxSize    : 0 +
                         (newOrigin[1] > 0) ?   newOrigin[1]*rStride[0] : 0 +
                         (newOrigin[2] > 0) ?   newOrigin[2]*rStride[1] : 0;
    size_t writeOffset = (newOrigin[0] < 0) ? - newOrigin[0]*voxSize    : 0 +
                         (newOrigin[1] < 0) ? - newOrigin[1]*wStride[0] : 0 +
                         (newOrigin[2] < 0) ? - newOrigin[2]*wStride[1] : 0;
    for (auto t = 0; t < copyRange[3]; t++)
    for (auto z = 0; z < copyRange[2]; z++)
    for (auto y = 0; y < copyRange[1]; y++)
    {
        auto r = readPtr  + y * rStride[0] + z * rStride[1] + t * rStride[2];
        auto w = writePtr + y * wStride[0] + z * wStride[1] + t * wStride[2];

        memcpy(w + writeOffset, r + readOffset, copyLength);

        // Clear the rest of the row to the setval
        // :TODO:
    }

    // Set the padded value data for the uncopied rows
    // :TODO:

    // Compute the time elapsed during execution
    auto tend = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(tend-tbeg);
    VOX_LOG_INFO(VOLT_LOG_CAT, format("Crop operation completed in %1% ms", time.count()));

    return Volume::create(data, newExtent, volume->spacing(), volume->offset(), volume->type());
}

} // namespace volt
} // namespace vox
