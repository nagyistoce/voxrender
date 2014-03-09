/* ===========================================================================

	Project: Standard Volume Filters
    
	Description: Exposes some standard filters provided by the volt library

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
#ifndef SVF_FILTERS_H
#define SVF_FILTERS_H

// Include Dependencies
#include "Plugins/StdVolumeFilter/Common.h"
#include "VoxLib/Core/CudaCommon.h"
#include "VoxLib/Error/Error.h"
#include "VoxVolt/Core.h"

// API namespace
namespace vox
{
   
/** Lanczos resize transformation provided by Volt library */
class LanczosFilter : public volt::Filter
{
public:
    LanczosFilter(std::shared_ptr<void> handle) : m_handle(handle) { }

    String name() { return "Filters.Sampling.Lanczos Resize"; }
    
    void getParams(std::list<volt::FilterParam> & params)
    {
    }

    void execute(Scene & scene, OptionSet const& params)
    {
    }

private:
    std::shared_ptr<void> m_handle;
};


/** Linear transformation provided by Volt library */
class LinearFilter : public volt::Filter
{
public:
    LinearFilter(std::shared_ptr<void> handle) : m_handle(handle) { }

    String name() { return "Filters.Sampling.Shift and Scale"; }
    
    void getParams(std::list<volt::FilterParam> & params)
    {
        params.push_back(volt::FilterParam("Shift", volt::FilterParam::Type_Float, "0", "[-10000.0 10000.0]"));
        params.push_back(volt::FilterParam("Scale", volt::FilterParam::Type_Float, "1.0", "[-10000.0 10000.0]"));
        params.push_back(volt::FilterParam("Type", volt::FilterParam::Type_Radio, "UInt8", "[UInt32 Int32 UInt16 Int16 UInt8 Int8 Float32 Float64]"));
    }

    void execute(Scene & scene, OptionSet const& params)
    {
        double shift = params.lookup<double>("Shift");
        double scale = params.lookup<double>("Scale");
        volt::Linear::execute(scene.volume, shift, scale);
    }

private:
    std::shared_ptr<void> m_handle;
};

/** Averaging filter provided by Volt library */
class MeanFilter : public volt::Filter
{
public:
    MeanFilter(std::shared_ptr<void> handle) : m_handle(handle) { }

    String name() { return "Filters.Convolution.Mean"; }
    
    void getParams(std::list<volt::FilterParam> & params)
    {
        params.push_back(volt::FilterParam("Kernel Size", volt::FilterParam::Type_Int, "0", "[3 10]"));
    }

    void execute(Scene & scene, OptionSet const& params)
    {
        auto variance = params.lookup<float>("Variance");
        auto size     = params.lookup<unsigned int>("Kernel Size");
        Image3D<float> kernel(size, size, size);
        std::vector<float> avgVec;
        volt::Conv::makeMeanKernel(avgVec, size);
        for (int x = 0; x < size; x++)
        for (int y = 0; y < size; y++)
        for (int z = 0; z < size; z++)
            kernel.at(x, y, z) = avgVec[x] * avgVec[y] * avgVec[z];
        scene.volume = volt::Conv::execute(scene.volume, kernel);
    }

private:
    std::shared_ptr<void> m_handle;
};

/** Gaussian filter provided by Volt library */
class GaussFilter : public volt::Filter
{
public:
    GaussFilter(std::shared_ptr<void> handle) : m_handle(handle) { }

    String name() { return "Filters.Convolution.Gaussian"; }
    
    void getParams(std::list<volt::FilterParam> & params)
    {
        params.push_back(volt::FilterParam("Variance", volt::FilterParam::Type_Float, "0.75", "[0.0 5.0]"));
        params.push_back(volt::FilterParam("Kernel Size", volt::FilterParam::Type_Int, "0", "[0 10]"));
    }

    void execute(Scene & scene, OptionSet const& params)
    {
        auto variance = params.lookup<float>("Variance");
        auto size     = params.lookup<unsigned int>("Kernel Size");
        std::vector<float> gaussVec;
        volt::Conv::makeGaussianKernel(gaussVec, variance, size);
        size = gaussVec.size();
        Image3D<float> kernel(size, size, size);
        for (int x = 0; x < size; x++)
        for (int y = 0; y < size; y++)
        for (int z = 0; z < size; z++)
            kernel.at(x, y, z) = gaussVec[x] * gaussVec[y] * gaussVec[z];
        scene.volume = volt::Conv::execute(scene.volume, kernel);
    }

private:
    std::shared_ptr<void> m_handle;
};

/** Laplacian filter provided by Volt library */
class LaplaceFilter : public volt::Filter
{
public:
    LaplaceFilter(std::shared_ptr<void> handle) : m_handle(handle) { }

    String name() { return "Filters.Convolution.Laplace"; }
    
    void getParams(std::list<volt::FilterParam> & params)
    {
    }

    void execute(Scene & scene, OptionSet const& params)
    {
        Image3D<float> kernel;
        volt::Conv::makeLaplaceKernel(kernel);
        scene.volume = volt::Conv::execute(scene.volume, kernel);
    }

private:
    std::shared_ptr<void> m_handle;
};

} // namespace vox

// End definition
#endif // SVF_FILTERS_H