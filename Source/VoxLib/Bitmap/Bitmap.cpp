/* ===========================================================================

	Project: VoxRender - Bitmap
    
	Description: Defines a generic image class

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
#include "Bitmap.h"

// Include Dependencies
#include "VoxLib/Core/Common.h"
#include "VoxLib/Core/Format.h"
#include "VoxLib/Core/Functors.h"
#include "VoxLib/Core/Logging.h"

namespace vox {

namespace {
namespace filescope {

    static std::map<String, std::shared_ptr<ImageImporter>> importers;   // Registered import modules
    static std::map<String, std::shared_ptr<ImageExporter>> exporters;   // Registered export modules 

    static boost::shared_mutex moduleMutex; // Module access mutex for read-write locks

} // namespace filescope
} // namespace anonymous

// --------------------------------------------------------------------
//  Registers a new resource import module
// --------------------------------------------------------------------
void Bitmap::registerImportModule(String const& extension, std::shared_ptr<ImageImporter> importer)
{ 
    // Acquire a read-lock on the modules for thread safety support
    boost::unique_lock<decltype(filescope::moduleMutex)> lock(filescope::moduleMutex);
    
    if (!importer) throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY, 
        "Attempted to register empty import module", Error_NotAllowed);

    filescope::importers[extension] = importer; 
}

// --------------------------------------------------------------------
//  Registers a new resource export module
// --------------------------------------------------------------------
void Bitmap::registerExportModule(String const& extension, std::shared_ptr<ImageExporter> exporter)
{ 
    // Acquire a read-lock on the modules for thread safety support
    boost::unique_lock<decltype(filescope::moduleMutex)> lock(filescope::moduleMutex);

    if (!exporter) throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY, 
        "Attempted to register empty export module", Error_NotAllowed);

    filescope::exporters[extension] = exporter; 
}

// --------------------------------------------------------------------
//  Removes a image import module
// --------------------------------------------------------------------
void Bitmap::removeImportModule(std::shared_ptr<ImageImporter> importer, String const& extension)
{
    // Acquire a read-lock on the modules for thread safety support
    boost::unique_lock<decltype(filescope::moduleMutex)> lock(filescope::moduleMutex);
    
    if (extension.empty())
    {
        auto iter = filescope::importers.begin();
        while (iter != filescope::importers.end())
        {
            if (iter->second == importer)
            {
                auto old = iter; ++iter;
                filescope::importers.erase(old);
            }
            else
            {
                ++iter;
            }
        }
    }
    else
    {
        auto iter = filescope::importers.find(extension);
        if (iter != filescope::importers.end()) 
            filescope::importers.erase(iter);
    }
}

// --------------------------------------------------------------------
//  Removes a image export module
// --------------------------------------------------------------------
void Bitmap::removeExportModule(std::shared_ptr<ImageExporter> exporter, String const& extension)
{
    // Acquire a read-lock on the modules for thread safety support
    boost::unique_lock<decltype(filescope::moduleMutex)> lock(filescope::moduleMutex);
    
    if (extension.empty())
    {
        auto iter = filescope::exporters.begin();
        while (iter != filescope::exporters.end())
        {
            if (iter->second == exporter)
            {
                auto old = iter; ++iter;
                filescope::exporters.erase(old);
            }
            else
            {
                ++iter;
            }
        }
    }
    else
    {
        auto iter = filescope::exporters.find(extension);
        if (iter != filescope::exporters.end()) 
            filescope::exporters.erase(iter);
    }
}

// ----------------------------------------------------------------------------
//  Imports a image using a matching registered importer
// ----------------------------------------------------------------------------
Bitmap Bitmap::imprt(ResourceIStream & data, OptionSet const& options)
{
    return imprt(data, data.identifier().extractFileExtension(), options);
}

// ----------------------------------------------------------------------------
//  Imports a image using a matching registered importer
// ----------------------------------------------------------------------------
Bitmap Bitmap::imprt(std::istream & data, String const& extension, OptionSet const& options)
{
    // Acquire a read-lock on the modules for thread safety support
    boost::shared_lock<decltype(filescope::moduleMutex)> lock(filescope::moduleMutex);

	// Execute the register import module
    auto importer = filescope::importers.find(extension);
    if (importer != filescope::importers.end())
    {
        return importer->second->importer(data, options);
    }

    throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY, 
                "No import module found", Error_BadToken);
}

// ----------------------------------------------------------------------------
//  Exports a image using a matching registered exporter
// ----------------------------------------------------------------------------
void Bitmap::exprt(ResourceOStream & data, OptionSet const& options) const
{
    exprt(data, data.identifier().extractFileExtension(), options);
}

// ----------------------------------------------------------------------------
//  Exports a image using a matching registered exporter
// ----------------------------------------------------------------------------
void Bitmap::exprt(std::ostream & data, String const& extension, OptionSet const& options) const
{
    // Acquire a read-lock on the modules for thread safety support
    boost::shared_lock<decltype(filescope::moduleMutex)> lock(filescope::moduleMutex);

	// Execute the register import module
    auto exporter = filescope::exporters.find(extension);
    if (exporter != filescope::exporters.end())
    {
        exporter->second->exporter(data, options, *this);
    }
    else
    {
        throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY, 
                    "No export module found", Error_BadToken);
    }
}

// ----------------------------------------------------------------------------
//  Initializes an image for a known image format
// ----------------------------------------------------------------------------
Bitmap::Bitmap(Format type, size_t width, size_t height, size_t bitDepth, size_t stride, std::shared_ptr<void> data) :
    m_format(type),
    m_width(width),
    m_height(height),
    m_buffer(data)
{
    m_depth = bitDepth ? bitDepth : 8;

    switch (type)
    {
    case Format_RGB: 
        m_channels = 3;
        break;
    case Format_RGBA:
    case Format_RGBX:
        m_channels = 4;
        break;
    case Format_Gray:
        m_channels = 1;
        break;
    case Format_GrayAlpha:
        m_channels = 2;
        break;
    default: 
        throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY, format("Unrecognized format: %1%", type));
    }

    if (data) 
    {
        m_stride = stride ? stride : m_width * m_channels * m_depth/8;
    }
    else pad(0, false);
}

// ----------------------------------------------------------------------------
//  Initializes an image for an unknown image format
// ----------------------------------------------------------------------------
Bitmap::Bitmap(size_t width, size_t height, size_t bitDepth, size_t nChannels, 
                   size_t stride, std::shared_ptr<void> data) :
    m_format(Format_Unknown),
    m_width(width),
    m_height(height),
    m_channels(nChannels),
    m_depth(bitDepth),
    m_buffer(data)
{
    if (!data) pad(0, false);
    else
    {
        m_stride = stride ? stride : m_width * m_depth/8 * m_channels;
    }
}

// ----------------------------------------------------------------------------
//  Adjusts the stride of an image
// ----------------------------------------------------------------------------
void Bitmap::pad(size_t newStride, bool copyData)
{
    if (newStride == m_stride && m_buffer != nullptr) return;

    if (m_width == 0 || m_height == 0) return;

    auto rowSize = m_width * m_channels * m_depth / 8;
    auto actualStride = (newStride == 0) ? rowSize + (4 - (rowSize%4)) : newStride;
    if (actualStride < rowSize)
        throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY, "Requested stride smaller than row size", Error_Range);

    if (copyData)
    {
        auto buffer   = makeSharedArray(m_stride * m_height);
        auto writePtr = (char*)buffer.get();
        auto readPtr  = (char*)m_buffer.get();
        for (size_t i = 0; i < m_height; i++)
        {
            memcpy(writePtr, readPtr, rowSize);
            writePtr += actualStride;
            readPtr  += m_stride;
        }

        m_buffer = buffer;
    }
    else 
    {
        m_buffer.reset(new UInt8[actualStride * m_height], arrayDeleter);
    }

    m_stride = actualStride;
}

} // namespace vox