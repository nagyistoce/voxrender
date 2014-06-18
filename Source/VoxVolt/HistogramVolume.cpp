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
#include "HistogramVolume.h"

// Include Dependencies
#include "VoxLib/Core/Logging.h"

// Std library
#include <chrono>

namespace vox {
namespace volt {

namespace {
namespace filescope {
    
    // ----------------------------------------------------------------------------
    //  Core method for the histogram volume generator
    //  :TODO: Won't handle float or double volumes well, but most other stuff 
    //         doesn't anyway
    // ----------------------------------------------------------------------------
    template <typename T, typename S>
    void computeHistogram(std::shared_ptr<Volume> & volume, 
                          std::shared_ptr<Volume> & histoVol,
                          float gradScale = 1.0f, 
                          float lapScale = 1.0f)
    {
#define VOX_SCALE(x) x = (((x + 0.5f) / static_cast<float>(std::numeric_limits<T>::max())) - range[0]) / range[1];
        
        auto extent = volume->extent();
        auto size   = histoVol->extent();
        auto skip    = extent[0] * extent[1];
        auto range   = volume->valueRange();
        range[1] -= range[0];
        auto dataPtr = (T*)volume->data();
	    for (size_t t = 0; t < extent[3]; t++)
	    for (size_t k = 0; k < extent[2]; k++)
	    for (size_t j = 0; j < extent[1]; j++)
	    for (size_t i = 0; i < extent[0]; i++)
        {
            // Coordinates:
            //    0 |6
            //  1 2 3
            // 5| 4

            // Extract the neighbor data points required for estimation
            auto c2 = static_cast<S>(*dataPtr);
            VOX_SCALE(c2)
            auto c1 = i ? static_cast<S>(*(dataPtr-1)) : c2;
            VOX_SCALE(c1)
            auto c3 = i == extent[0]-1 ? c2 : static_cast<S>(*(dataPtr+1));  
            VOX_SCALE(c3)

            auto c4 = j ? static_cast<S>(*(dataPtr-extent[0])) : c2;
            VOX_SCALE(c4)
            auto c0 = (j == extent[1]-1) ? c2 : static_cast<S>(*(dataPtr+extent[0]));
            VOX_SCALE(c0)
            
            auto c5 = k ? static_cast<S>(*(dataPtr-skip)) : c2;
            VOX_SCALE(c5)
            auto c6 = (k == extent[2]-1) ? c2 : static_cast<S>(*(dataPtr+skip));
            VOX_SCALE(c6)

            // Compute 1st order gradient
            auto xd = c3 - c1;
            auto yd = c0 - c4;
            auto zd = c6 - c5;
            auto gradient = sqrt(xd*xd + yd*yd + zd*zd) / sqrt(3);
            gradient = low<S>(gradient*gradScale, 1.0);
        
            // Compute 2nd order gradient
            auto twoC = - (c2+c2);
            auto xl = c3 + c1 + twoC;
            auto yl = c0 + c4 + twoC;
            auto zl = c5 + c6 + twoC;
            auto laplace = sqrt(xl*xl + yl*yl + zl*zl) / sqrt(3);
            laplace = low<S>(laplace*lapScale, 1.0);

            // Update the histogram volume
            auto xpos = clamp<size_t>(c2*(size[0]-1), 0, size[0]-1);
            auto ypos = clamp<size_t>(gradient*(size[1]-1), 0, size[1]-1);
            auto zpos = clamp<size_t>(laplace*(size[2]-1), 0, size[2]-1);
            auto & voxel = *(UInt8*)histoVol->at(xpos, ypos, zpos, 0);
            if (voxel < std::numeric_limits<UInt8>::max()) ++voxel;

            // Increment the data ptr
            ++dataPtr;
        }
#undef VOX_SCALE
    }

    // ----------------------------------------------------------------------------
    //  Computes the range of gradient and laplace for a histogram volume which
    //  will encompass 100*(1-outlier) percent of the data points
    // ----------------------------------------------------------------------------
    void computeHistoRange(std::shared_ptr<Volume> volume,
                           size_t cutoff, 
                           float & gradScale, 
                           float & lapScale)
    {
        auto extent = volume->extent();
        
        size_t ycut = extent[1] - 1;
        size_t zcut = extent[2] - 1;
        
        auto yskip = extent[0];
        auto zskip = yskip*extent[1];
        auto tskip = zskip*extent[2];

        // Determine the gradient cutoff along the Y axis
        bool done = false;
        size_t hits = 0;
        while (ycut)
        {
            auto dataPtr = (UInt8*)volume->data() + ycut*yskip;
            for (size_t t = 0; t < extent[3]; t++)
            for (size_t z = 0; z < extent[2]; z++)
            for (size_t x = 0; x < extent[0]; x++)
            {
                auto val = *(dataPtr + x + z*zskip + t*tskip);
                if (val == std::numeric_limits<UInt8>::max()) 
                    done = true;
                else hits += val;
            }

            if (done || hits >= cutoff) break;

            ycut--;
        }
        
        // Determine the laplace cutoff along the Y axis
        done = false;
        hits = 0;
        while (zcut)
        {
            auto dataPtr = (UInt8*)volume->data() + zcut*zskip;
            for (size_t t = 0; t < extent[3]; t++)
            for (size_t y = 0; y < extent[1]; y++)
            for (size_t x = 0; x < extent[0]; x++)
            {
                auto val = *(dataPtr + x + y*yskip + t*tskip);
                if (val == std::numeric_limits<UInt8>::max()) 
                    done = true;
                else hits += val;
            }

            if (done || hits >= cutoff) break;

            zcut--;
        }

        // Convert the cutoffs to a gradient values
        auto gradCut = static_cast<float>(ycut) / static_cast<float>(extent[1]-1);
        gradScale = 1.0f / gradCut;
        auto lapCut  = static_cast<float>(zcut) / static_cast<float>(extent[2]-1);
        lapScale = 1.0f / lapCut;

        VOX_LOG_DEBUG(VOLT_LOG_CAT, format("Histogram cutoffs: gradient=%1% laplace=%2%", gradCut, lapCut));
    }

} // namespace filescope
} // namespace anonymous

// ----------------------------------------------------------------------------
//  Generates a histogram volume from a volume data set
// ----------------------------------------------------------------------------
std::shared_ptr<Volume> HistogramVolume::build(
    std::shared_ptr<Volume> volume, 
    Vector3s const& size,
    OptionSet & options
    )
{
    auto tbeg = std::chrono::high_resolution_clock::now();

    auto histoVol = Volume::create(nullptr, Vector4s(size[0], size[1], size[2], 1), Vector4f(1.0f), Vector3f(0), Volume::Type_UInt8);
    
    if (volume->extent().fold(mul) == 0) return histoVol;

    // Compute a histogram for the maximum gradient/laplace value ranges
    switch(volume->type())
    {
    case Volume::Type_UInt8: filescope::computeHistogram<UInt8,float>(volume, histoVol); break;
    case Volume::Type_Int8: filescope::computeHistogram<Int8,float>(volume, histoVol); break;
    case Volume::Type_UInt16: filescope::computeHistogram<UInt16,float>(volume, histoVol); break;
    case Volume::Type_Int16: filescope::computeHistogram<Int16,float>(volume, histoVol); break;
    case Volume::Type_UInt32: filescope::computeHistogram<UInt32,float>(volume, histoVol); break;
    case Volume::Type_Int32: filescope::computeHistogram<Int32,float>(volume, histoVol); break;
    default:
        throw Error(__FILE__, __LINE__, VOLT_LOG_CAT,
            "Unsupported volume format", Error_NotImplemented);
    }

    // Compute the gradient/laplace range for the final pass
    float gradScale;
    float lapScale;
    size_t cutoff = 0.05f * volume->extent().fold(&mul);
    filescope::computeHistoRange(histoVol, cutoff, gradScale, lapScale);

    // Compute the final histogram volume
    memset(histoVol->mutableData(), 0, histoVol->extent().fold(&mul));
    switch(volume->type())
    {
    case Volume::Type_UInt8: filescope::computeHistogram<UInt8,float>(volume, histoVol, gradScale, lapScale); break;
    case Volume::Type_Int8: filescope::computeHistogram<Int8,float>(volume, histoVol, gradScale, lapScale); break;
    case Volume::Type_UInt16: filescope::computeHistogram<UInt16,float>(volume, histoVol, gradScale, lapScale); break;
    case Volume::Type_Int16: filescope::computeHistogram<Int16,float>(volume, histoVol, gradScale, lapScale); break;
    case Volume::Type_UInt32: filescope::computeHistogram<UInt32,float>(volume, histoVol, gradScale, lapScale); break;
    case Volume::Type_Int32: filescope::computeHistogram<Int32,float>(volume, histoVol, gradScale, lapScale); break;
    default:
        throw Error(__FILE__, __LINE__, VOLT_LOG_CAT,
            "Unsupported volume format", Error_NotImplemented);
    }

    auto tend = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(tend-tbeg);
    
    VOX_LOG_INFO(VOLT_LOG_CAT, format("Histogram volume generation time: %1%", time.count()));

    return histoVol;
}

} // namespace volt
} // namespace vox
