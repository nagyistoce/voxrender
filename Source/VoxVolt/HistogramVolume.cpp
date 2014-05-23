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

// ----------------------------------------------------------------------------
//  Generates a histogram volume from a volume data set
//  :TODO: Most of this could probably be heavily optimized by loop unrolling
// ----------------------------------------------------------------------------
std::shared_ptr<Volume> HistogramVolume::build(std::shared_ptr<Volume> volume, Vector3s const& size)
{
    auto tbeg = std::chrono::high_resolution_clock::now();

    auto histoVol = Volume::create(nullptr, Vector4s(size[0], size[1], size[2], 1), Vector4f(1.0f), Vector3f(0), Volume::Type_UInt8);

    auto extent = volume->extent();
    
    if (extent.fold(mul) == 0) return histoVol;
    /*
    // (1) Calculate volume statistics
    Vector3f fmax(0.0f);
    Vector3f favg(0.0f);
    Vector3f fvar(0.0f);
    float N = 0.0f;
    for (size_t t = 0; t < extent[3]; t++)
	for (size_t k = 0; k < extent[2]; k++)
	for (size_t j = 0; j < extent[1]; j++)
	for (size_t i = 0; i < extent[0]; i++)
    {
        // Extract the neighbor data points required for estimation
        auto c = volume->fetchNormalized(i, j, k );

        auto im = i ? i-1 : 0;
        auto ip = low<size_t>(i+1, extent[0]-1);
        auto x0 = volume->fetchNormalized(im, j, k); 
        auto x2 = volume->fetchNormalized(ip, j, k);
        
        auto jm = j ? j-1 : 0;
        auto jp = low<size_t>(j+1, extent[1]-1);
        auto y0 = volume->fetchNormalized(i, jm, k);
        auto y2 = volume->fetchNormalized(i, jp, k);
        
        auto km = k ? k-1 : 0;
        auto kp = low<size_t>(k+1, extent[2]-1);
        auto z0 = volume->fetchNormalized(i, j, km);
        auto z2 = volume->fetchNormalized(i, j, kp);

        auto xd = x2 - x0;
        auto yd = y2 - y0;
        auto zd = z2 - z0;
        auto gradient = sqrt(xd*xd + yd*yd + zd*zd) / sqrt(3);
        auto gradDist = gradient - favg[1];
        if (gradient > fmax[1]) fmax[1] = gradient;
        favg[1] = (favg[1] * N + gradient) / (N + 1.0f);
        fvar[1] += (gradient - favg[1]) * gradDist;

        auto xl = x2 + x0 - c * 2;
        auto yl = y2 + y0 - c * 2;
        auto zl = z2 + z0 - c * 2;
        auto laplace = sqrt(xl*xl + yl*yl + zl*zl) / sqrt(3);
        auto lapDist = laplace - favg[2];
        if (laplace > fmax[2]) fmax[2] = laplace;
        favg[2] = (favg[2] * N + laplace) / (N + 1.0f);
        fvar[2] += (laplace - favg[2]) * lapDist;

        ++N;
    }

    fvar.map([=] (float x) { return sqrt(x / (N - 1.0f)); });
    auto minF = favg - fvar * 1.96;
    auto maxF = favg + fvar * 1.96;*/

    // (2) Generate the histogram volume
	for (size_t t = 0; t < extent[3]; t++)
	for (size_t k = 0; k < extent[2]; k++)
	for (size_t j = 0; j < extent[1]; j++)
	for (size_t i = 0; i < extent[0]; i++)
    {
        // Extract the neighbor data points required for estimation
        auto c = volume->fetchNormalized(i, j, k);

        auto im = i ? i-1 : 0;
        auto ip = low<size_t>(i+1, extent[0]-1);
        auto x0 = volume->fetchNormalized(im, j, k); 
        auto x2 = volume->fetchNormalized(ip, j, k);
        
        auto jm = j ? j-1 : 0;
        auto jp = low<size_t>(j+1, extent[1]-1);
        auto y0 = volume->fetchNormalized(i, jm, k);
        auto y2 = volume->fetchNormalized(i, jp, k);
        
        auto km = k ? k-1 : 0;
        auto kp = low<size_t>(k+1, extent[2]-1);
        auto z0 = volume->fetchNormalized(i, j, km);
        auto z2 = volume->fetchNormalized(i, j, kp);
        
        // Compute estimated density, gradient, and laplacian
        auto density  = volume->fetchNormalized(i, j, k);

        auto xd = x2 - x0;
        auto yd = y2 - y0;
        auto zd = z2 - z0;
        auto gradient = sqrt(xd*xd + yd*yd + zd*zd) / sqrt(3);
        
        auto xl = x2 + x0 - c * 2;
        auto yl = y2 + y0 - c * 2;
        auto zl = z2 + z0 - c * 2;
        auto laplace = sqrt(xl*xl + yl*yl + zl*zl) / sqrt(3);

        // Update the histogram volume
        auto xpos = clamp<size_t>(density*(size[0]-1), 0, size[0]-1);
        auto ypos = size[1]-1-clamp<size_t>(gradient*(size[1]-1), 0, size[1]-1);
        auto zpos = size[2]-1-clamp<size_t>(laplace*(size[2]-1), 0, size[2]-1);
        auto & voxel = *(UInt8*)histoVol->at(xpos, ypos, zpos, 0);
        if (voxel < std::numeric_limits<UInt8>::max()) ++voxel;
    }

    auto tend = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(tend-tbeg);
    
    VOX_LOG_INFO(VOLT_LOG_CAT, format("Histogram volume generation time: %1%", time.count()));

    return histoVol;
}

} // namespace volt
} // namespace vox
