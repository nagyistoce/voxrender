/* ===========================================================================

	Project: VoxRender - Vox Transfer File
    
	Description: Vox transfer file import/export module

    Copyright (C) 2012 Lucas Sherman

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
#include "VoxTransferFile.h"

// Include Dependencies
#include "VoxLib/Core/Functors.h"
#include "VoxLib/Core/Format.h"
#include "VoxLib/Core/Logging.h"
#include "VoxLib/Scene/Volume.h"

// API namespace
namespace vox
{

// --------------------------------------------------------------------
//  Writes a transfer function XML to the output resource
// --------------------------------------------------------------------
void VoxTransferFile::exporter(ResourceOStream & data, 
                               OptionSet const&  options, 
                               Scene const&      scene)
{
    // Detect missing volume
    if (!scene.volume)
    {
        Logger::addEntry(
            Severity_Warning, Error_MissingData, VOX_LOG_CATEGORY,
            "No volume data in scene", __FILE__, __LINE__);
    }

    // :TODO: byte ordering option with Boost.Endian library
    // :TODO: compression with bzip, bzip2, zlib

    // Output raw volume data to the data stream
    size_t size = scene.volume->extent().fold<size_t>(1, &vox::mul);
    data.write(reinterpret_cast<char const*>(scene.volume->data()), size); 
}

// --------------------------------------------------------------------
//  Reads a raw volume object from the stream
// --------------------------------------------------------------------
Scene VoxTransferFile::importer(ResourceIStream & data, OptionSet const& options)
{
    auto & volume = *std::make_shared<Volume>();

    // Ensure size information is available
    auto iter = options.find("Size");
    if (iter != options.end())
    {
        throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY,
                    "Raw volume import requires \"Size\" option.",
                    Error_MissingData);
    }

    // :TODO: byte ordering option with Boost.Endian library
    // :TODO: compression with bzip, bzip2, zlib

    // Read data from the resource stream
    auto extent = boost::lexical_cast<Vector<size_t,3>>((*iter).first);
    size_t bytes = extent.fold<size_t>(1, &vox::mul); 
    std::shared_ptr<UInt8> rawData(new UInt8[bytes], &arrayDeleter);
    data.read(reinterpret_cast<char*>(rawData.get()), bytes);
    volume.setData(rawData, extent.resize<4>()); // :TODO: Remove 4th dimension flag

    return Scene();
}

}