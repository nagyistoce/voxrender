/* ===========================================================================

	Project: VoxRender

	Description: Generates the histogram images for the volume data

    Copyright (C) 2013 Lucas Sherman

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
#include "histogramgenerator.h"

// Include Dependencies
#include "VoxScene/Volume.h"
#include "VoxLib/Core/Logging.h"
#include "VoxVolt/HistogramVolume.h"

#define R3I 0.57735026918962576450914878050196f;

using namespace vox;

namespace {
namespace filescope {

    // --------------------------------------------------------------------
    // Iterates over the volume voxels and performs the histogram binning
    // --------------------------------------------------------------------
    template<typename T> std::vector<size_t> generateHistogramBins(size_t nBins, std::shared_ptr<Volume> volume)
    {
        Vector2f const& range    = volume->valueRange();
        size_t          elements = volume->extent().fold<size_t>(1, &mul);
        T const*        data     = reinterpret_cast<T const*>(volume->data());
        float           max      = static_cast<float>(std::numeric_limits<T>::max());

        std::vector<size_t> bins(nBins, 0);

        auto extent = volume->extent();
	    for (size_t k = 0; k < extent[2]; k++)
	    for (size_t j = 0; j < extent[1]; j++)
	    for (size_t i = 0; i < extent[0]; i++)
        {
            float sample = volume->fetchNormalized(i, j, k);

            size_t bin = clamp<size_t>(static_cast<size_t>(sample*(nBins-1)), 0, nBins-1);

            bins[bin]++;
        }

        return bins;
    }

    // ----------------------------------------------------------------------------
    //  Generates histogram information for the volume
    // ----------------------------------------------------------------------------
    std::vector<size_t> generateBins(std::shared_ptr<Volume> volume)
    {
        switch (volume->type())
        {
            case Volume::Type_UInt8:  
            {
                size_t min = 255 * volume->valueRange()[0];
                size_t max = 255 * volume->valueRange()[1];
                return filescope::generateHistogramBins<UInt8>(max-min+1, volume);
            }

            case Volume::Type_UInt16:
            {
                return filescope::generateHistogramBins<UInt16>(512, volume);
            }
            
            case Volume::Type_Int16:
            {
                return filescope::generateHistogramBins<UInt16>(512, volume);
            }

            default:
                throw Error(__FILE__, __LINE__, VSR_LOG_CATEGORY,
                    format("Unsupported volume data type (%1%)", 
                           Volume::typeToString(volume->type())),
                    Error_NotImplemented);
        }
    }

} // namespace filescope
} // namespace anonymous

// ----------------------------------------------------------------------------
//  Initialize some slot-signal connections
// ----------------------------------------------------------------------------
HistogramGenerator::HistogramGenerator() 
{
    connect(MainWindow::instance, SIGNAL(sceneChanged(vox::Scene &,void *)), 
            this, SLOT(onSceneChanged(vox::Scene &,void *)), Qt::DirectConnection);

    connect(this, SIGNAL(histogramImageReady(int)), this, SLOT(onHistogramImageReady(int)));
}

// ----------------------------------------------------------------------------
//  Terminates the historgram generation thread
// ----------------------------------------------------------------------------
HistogramGenerator::~HistogramGenerator() 
{
    stopGeneratingImages();
}

// ----------------------------------------------------------------------------
//  Converts the final histogram image into a pixmap and frees it from memory
// ----------------------------------------------------------------------------
void HistogramGenerator::onHistogramImageReady(int dataType)
{
    switch (dataType)
    {
    case HistogramView::DataType_Density:
        {
        auto densityImage = QImage(reinterpret_cast<uchar*>(m_densityImage.data()), 
            m_densityImage.width(), m_densityImage.height(), m_densityImage.stride(), 
            QImage::Format_ARGB32);
        m_densityPixmap.convertFromImage(densityImage);
        }
        break;
    case HistogramView::DataType_DensityGrad:
        {
        auto gradientImage = QImage(reinterpret_cast<uchar*>(m_gradientImage.data()), 
            m_gradientImage.width(), m_gradientImage.height(), m_gradientImage.stride(), 
            QImage::Format_ARGB32);
        m_gradientPixmap.convertFromImage(gradientImage);
        }
        break;
    case HistogramView::DataType_DensityLap:
        {
        auto laplaceImage = QImage(reinterpret_cast<uchar*>(m_laplaceImage.data()), 
            m_laplaceImage.width(), m_laplaceImage.height(), m_laplaceImage.stride(), 
            QImage::Format_ARGB32);
        m_laplacePixmap.convertFromImage(laplaceImage);
        }
        break;
    default:
        VOX_LOG_ERROR(Error_Bug, "GUI", "Unrecognized histogram type");
        return;
    }

    emit histogramComplete(dataType);
}

// ----------------------------------------------------------------------------
//  Entry point for histogram image generation thread
// ----------------------------------------------------------------------------
void HistogramGenerator::threadEntryPoint(std::shared_ptr<vox::Volume> volume)
{
    generateDensityHistogram(volume);
    
    auto histoVol = volt::HistogramVolume::build(volume);

    generateGradientHistogram(histoVol);
    generateLaplacianHistogram(histoVol);
}

// ----------------------------------------------------------------------------
//  Draws a density histogram of the specified volume data set
// ----------------------------------------------------------------------------
void HistogramGenerator::generateDensityHistogram(std::shared_ptr<vox::Volume> volume)
{
    auto tbeg = std::chrono::high_resolution_clock::now();

    // Extracted binned density information from the volume
    auto bins   = filescope::generateBins(volume);
    auto binMax = bins[0];
    BOOST_FOREACH (auto & bin, bins)
    {
        if (bin > binMax) binMax = bin;
    }

    // Draw the histogram image
    m_densityImage.resize(bins.size(), bins.size()/2);

    memset(m_densityImage.data(), 0, m_densityImage.stride()*m_densityImage.height());

    auto const max     = logf(1.0f+binMax);
    auto const height  = m_densityImage.height();
    auto const width   = m_densityImage.width();
	auto const wscalar = bins.size() / width;
    auto const hscalar = height / max;

	for (size_t i = 0; i < width; i++)
	{
		size_t bucket    = static_cast<size_t>(i*wscalar);
		size_t maxHeight = height - static_cast<size_t>(logf(1.0f+bins[bucket]) * hscalar);

		for (size_t j = maxHeight; j < height; j++)
		{
            auto & pixel = m_densityImage.at(i, j);
            pixel.b = 234;
            pixel.g = 217;
            pixel.r = 153;
            pixel.a = 160;
		}
	}
    
    auto tend = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(tend-tbeg);

    VOX_LOG_INFO("GUI", format("Density histogram generation time: %1%", time.count()));

    emit histogramImageReady(HistogramView::DataType_Density);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
void HistogramGenerator::generateGradientHistogram(std::shared_ptr<vox::Volume> volume)
{
    auto tbeg = std::chrono::high_resolution_clock::now();

    auto extent = volume->extent();
    m_gradientImage.resize(extent[0], extent[1]);
    m_gradientImage.clear();

    vox::Image<unsigned long> bins(extent[0], extent[1]);
    bins.clear(0);

    // Collapse the volume into the mag-grad range
    auto dataPtr = volume->data();
    for (size_t k = 0; k < extent[2]; ++k)
    for (size_t j = 0; j < extent[1]; ++j)
    for (size_t i = 0; i < extent[0]; ++i)
    {
        bins.at(i,j) += *(dataPtr++);
    }
    
    // Compute the bounds on the counts
    unsigned long cmax = 0;
	for (size_t j = 0; j < extent[1]; j++)
	for (size_t i = 0; i < extent[0];  i++)
    {
        auto count = bins.at(i, j);
        if (cmax < count) cmax = count;
    }
    
    // Compute the image pixel colors
    float logMax = log((double)cmax);
	for (size_t j = 0; j < extent[1]; j++)
	for (size_t i = 0; i < extent[0]; i++)
    {
        auto count = bins.at(i, j);
        if (count < 10) 
        {
            m_gradientImage.at(i, extent[1]-1-j).a = 0;
        }
        else
        {
            UInt8 normVal = (UInt8)(255.0 * (log((double)count) / logMax));
            m_gradientImage.at(i, extent[1]-1-j).a = vox::clamp<UInt8>(normVal, 0, 255);
        }
    }

    auto tend = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(tend-tbeg);

    VOX_LOG_INFO("GUI", format("Gradient histogram generation time: %1%", time.count()));

    emit histogramImageReady(HistogramView::DataType_DensityGrad);
}
        
// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
void HistogramGenerator::generateLaplacianHistogram(std::shared_ptr<vox::Volume> volume)
{
    auto tbeg = std::chrono::high_resolution_clock::now();
    auto extent = volume->extent();
    m_laplaceImage.resize(extent[0], extent[1]);
    m_laplaceImage.clear();

    vox::Image<unsigned long> bins(extent[0], extent[2]);
    bins.clear(0);

    // Collapse the volume into the mag-grad range
    auto dataPtr = volume->data();
    for (size_t k = 0; k < extent[2]; ++k)
    for (size_t j = 0; j < extent[1]; ++j)
    for (size_t i = 0; i < extent[0]; ++i)
    {
        bins.at(i,k) += *(dataPtr++);
    }
    
    // Compute the bounds on the counts
    unsigned long cmax = 0;
	for (size_t j = 0; j < extent[2]; j++)
	for (size_t i = 0; i < extent[0];  i++)
    {
        auto count = bins.at(i, j);
        if (cmax < count) cmax = count;
    }
    
    // Compute the image pixel colors
    float logMax = log((double)cmax);
	for (size_t j = 0; j < extent[2]; j++)
	for (size_t i = 0; i < extent[0]; i++)
    {
        auto count = bins.at(i, j);
        if (count < 10) 
        {
            m_laplaceImage.at(i, extent[1]-1-j).a = 0;
        }
        else
        {
            UInt8 normVal = (UInt8)(255.0 * (log((double)count) / logMax));
            m_laplaceImage.at(i, extent[1]-1-j).a = vox::clamp<UInt8>(normVal, 0, 255);
        }
    }

    auto tend = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(tend-tbeg);

    VOX_LOG_INFO("GUI", format("Laplacian histogram generation time: %1%", time.count()));

    emit histogramImageReady(HistogramView::DataType_DensityLap);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
void HistogramGenerator::onSceneChanged(Scene & scene, void * userInfo)
{
    if (!scene.volume || !scene.volume->isDataDirty()) return;

    stopGeneratingImages();

    m_thread = std::shared_ptr<boost::thread>( 
        new boost::thread(std::bind(&HistogramGenerator::threadEntryPoint, this, scene.volume)));
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
void HistogramGenerator::stopGeneratingImages()
{
    if (m_thread)
    {
        m_thread->interrupt();
        m_thread->join(); 
        m_thread.reset();
    }
}