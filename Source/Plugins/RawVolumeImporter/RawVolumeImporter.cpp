/* ===========================================================================

	Project: VoxRender - Raw Volume File
    
	Description: Raw volume file import/export module

    Copyright (C) 2012-2014 Lucas Sherman

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
#include "RawVolumeImporter.h"

// Include Dependencies
#include "VoxLib/Core/Common.h"
#include "VoxScene/Volume.h"
#include "VoxLib/Core/Functors.h"
#include "VoxLib/Core/Logging.h"

// API namespace
namespace vox
{

// File scope namespace
namespace
{
    namespace filescope
    {
        // --------------------------------------------------------------------
        //  Attempts to deduce a volume type from an input string
        // --------------------------------------------------------------------
        Volume::Type stringToType(String const& typeStr)
        {
            for (size_t t = Volume::Type_Begin; t != Volume::Type_End; t++)
            {
                if (typeStr == Volume::typeToString((Volume::Type)t)) return (Volume::Type)t;
            }

            throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY,
                format("Unrecognized volume data type (%1%)", typeStr),
                Error_BadToken);
        }

        // Export module implementation
        class RawExporter
        {
        public:
            // --------------------------------------------------------------------
            //  Parse the scene data into a boost::property_tree
            // --------------------------------------------------------------------
            RawExporter(ResourceOStream & sink, OptionSet const& options, Scene const& scene) :
                m_sink(sink), m_options(options), m_scene(scene)
            {
                // Compose the resource identifier for log warning entries
                std::string filename = sink.identifier().extractFileName();
                m_displayName = filename.empty() ? "UNKNOWN" : format("\"%1%\"", filename);
            }

            // --------------------------------------------------------------------
            //  Write the boost::property_tree as an XML file to the stream
            // --------------------------------------------------------------------
            void writeRawDataFile()
            {
                // Extract the option flags for data compression
                auto chain = boost::iostreams::filtering_streambuf<boost::iostreams::output>();

                // Detect the endian mode settings
                std::string endianess = m_options.lookup("Endianess", "little");
                boost::algorithm::to_lower(endianess);
                bool isLittleEndian;
                if (endianess == "little" || endianess == "lsb")   isLittleEndian = true;
                else if (endianess == "big" || endianess == "msb") isLittleEndian = false;
                else throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY,
                                format("Invalid Endianess: %1%", endianess),
                                Error_BadToken);

                // Determine if a swap operation is necessary
                bool swap =
                #ifdef BOOST_LITTLE_ENDIAN
                    ( (endianess == "big" || endianess == "msb"));
                #else
                    ( (endianess == "little" || endianess == "lsb"));
                #endif
                
                // Detect the compression mode options 
                std::vector<std::string> filters;
                auto compressionIter = m_options.find("Compression");
                if (compressionIter != m_options.end())
                {
                    boost::algorithm::split(
                        filters, 
                        compressionIter->second, 
                        boost::is_any_of(" ,\n\t\r"), 
                        boost::algorithm::token_compress_on
                        );
                }
                
                // Build the decompression filter chain in reverse order
                BOOST_FOREACH(auto & filter, filters)
                {
                    if (filter.empty()) { }
                    else if (filter == "zlib")  chain.push(boost::iostreams::zlib_compressor());
                    else if (filter == "gzip")  chain.push(boost::iostreams::gzip_compressor());
                    else if (filter == "bzip2") chain.push(boost::iostreams::bzip2_compressor());
                    else
                    {
                        throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY,
                                    format("Unsupported compression format: %1%", filter),
                                    vox::Error_BadToken);
                    }
                }

                // Attach the Device
                chain.push(m_sink);

                // Output raw volume data in the specified endian format
                size_t bytesPerVoxel = Volume::typeToSize(m_scene.volume->type());
                auto bytesToWrite    = m_scene.volume->extent().fold<size_t>(1, &mul) * bytesPerVoxel;
                auto bytesWritten    = boost::iostreams::write(chain, 
                    reinterpret_cast<char const*>(m_scene.volume->data()), bytesToWrite);
            }

        private:
            static int const versionMajor = 0; ///< Scene version major [potentially breaking]
            static int const versionMinor = 0; ///< Scene version minor [usually non-breaking]

            ResourceOStream & m_sink;        ///< Resource stream
            OptionSet const&  m_options;     ///< Import options
            Scene const&      m_scene;       ///< Scene data
            std::string       m_displayName; ///< Warning identifier
        };

        // Import module implementation
        class RawImporter
        {
        public:
            // --------------------------------------------------------------------
            //  Parse the resource data from an XML format into a property tree
            // --------------------------------------------------------------------
            RawImporter(ResourceIStream & source, OptionSet const& options, std::shared_ptr<void> & handle) : 
              m_options(options), m_source(source), m_handle(handle)
            {
                // Compose the resource identifier for log warning entries
                std::string filename = source.identifier().extractFileName();
                m_displayName = filename.empty() ? "UNKNOWN" : format("\"%1%\"", filename);
            }
            
            // --------------------------------------------------------------------
            //  Null operation, there is nothing to parse here
            // --------------------------------------------------------------------
            Scene readRawDataFile()
            {
                // Extract the required volume data size parameters
                auto typeStr = m_options.lookup<String>("Type");
                auto size    = m_options.lookup<Vector4u>("Size"); 
				auto spacing = m_options.lookup("Spacing", Vector4f(1.0f, 1.0f, 1.0f, 1.0f));
                auto offset  = m_options.lookup("Offset", Vector3f(0.0f, 0.0f, 0.0f));

                // Determine the volume data type
                boost::to_lower(typeStr);
                Volume::Type type = stringToType(typeStr);
                auto bytesPerVoxel = Volume::typeToSize(type);
                
                // Detect the endian mode settings and determine if we must swap
                bool swap = false;
                std::string endianess = m_options.lookup("Endianess", "little");
                boost::algorithm::to_lower(endianess);
                if (bytesPerVoxel > 1)
                if (endianess == "little" || endianess == "lsb")
                {
                #ifndef BOOST_LITTLE_ENDIAN
                    swap = true;
                #endif
                }
                else if (endianess == "big" || endianess == "msb")
                {
                #ifdef BOOST_LITTLE_ENDIAN
                    swap = true;
                #endif
                }
                else throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY,
                    format("Invalid Endianess: %1%", endianess), Error_BadToken);

                // Allocate memory for the volume data
                auto voxels = size.fold(mul);
                auto data = makeSharedArray(voxels*bytesPerVoxel);

                // Check if we are reading the first of a multi-file series
                // :NOTE: This is 1 file per volume, NOT 1 file per slice
                auto multiFile = m_options.lookup("MultiFile", "");
                if (!multiFile.empty())
                {
                    auto dataPtr = data.get();
                    auto slice = voxels / size[3]; // Time slice size 
                    auto range = m_options.lookup<Vector2u>("Range");
                    auto place = m_options.lookup<String>("Place");
                    for (unsigned int i = range[0]; i <= range[1]; i++)
                    {
                        auto tag = boost::lexical_cast<String>(i);
                        auto relative = multiFile;
                        boost::algorithm::replace_first(relative, place, tag);
                        ResourceId id = m_source.identifier().applyRelativeReference(relative);
                        m_source.close();
                        m_source.open(id);
                        readInputData(bytesPerVoxel, slice, dataPtr);
                        dataPtr += slice;
                    }
                }
                else // Single file raw load
                {
                    // Read the raw volume data from the filter chain
                    readInputData(bytesPerVoxel, voxels, data.get());
                }

                // Convert to the native byte ordering
                if (swap)
                {
                    UInt16 * ptr = (UInt16*)data.get();
                    for (size_t i = 0; i < voxels; i++)
                        ptr[i] = VOX_SWAP16(ptr[i]);
                }

                // Construct the volume object for the response
                Scene scene;
                scene.volume = Volume::create(data, size, spacing, offset, type);

                return scene;
            }
            
        private:
            // --------------------------------------------------------------------
            //  Constructs the filter for processing the formatted file data
            // --------------------------------------------------------------------
            inline void readInputData(size_t bpv, size_t voxels, UInt8* data)
            {
                auto chain = boost::iostreams::filtering_streambuf<boost::iostreams::input>();
                
                // Detect the compression mode options 
                std::vector<std::string> filters;
                auto compressionIter = m_options.find("Compression");
                if (compressionIter != m_options.end())
                {
                    boost::algorithm::split(
                        filters, 
                        compressionIter->second, 
                        boost::is_any_of(" ,\n\t\r"), 
                        boost::algorithm::token_compress_on
                        );
                }
                
                // Build the decompression filter chain in reverse order
                BOOST_FOREACH(auto & filter, filters)
                {
                    if (filter.empty()) { }
                    else if (filter == "zlib")  chain.push(boost::iostreams::zlib_decompressor());
                    else if (filter == "gzip")  chain.push(boost::iostreams::gzip_decompressor());
                    else if (filter == "bzip2") chain.push(boost::iostreams::bzip2_decompressor());
                    else
                    {
                        throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY,
                                    format("Unsupported compression format: %1%", filter),
                                    vox::Error_BadToken);
                    }
                }

                // Attach the Device
                chain.push(m_source);

                // Read the data through the finalized filter
                auto bytesToRead = bpv*voxels;
                auto bytesRead = boost::iostreams::read(
                    chain, reinterpret_cast<char*>(data), bytesToRead);
                if (bytesRead != bytesToRead)
                {
                    throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY,
                                format("Unable to load \"%1%\": expected %2% bytes, read %3% bytes", 
                                       m_displayName, bytesToRead, bytesRead),
                                Error_MissingData);
                }
            }

            static int const versionMajor = 0; ///< Version major
            static int const versionMinor = 0; ///< Version minor 
            
            std::shared_ptr<void> & m_handle;   ///< Plugin handle to track this DLL's usage

            ResourceIStream & m_source;       ///< Resource stream
            OptionSet const&  m_options;      ///< Import options
            std::string       m_displayName;  ///< Warning identifier
        };
    }
}

// --------------------------------------------------------------------
//  Writes a raw volume file to the stream
// --------------------------------------------------------------------
void RawVolumeFile::exporter(ResourceOStream & sink, OptionSet const& options, Scene const& scene)
{
    // Parse scenefile object into boost::property_tree
    filescope::RawExporter exportModule(sink, options, scene);

    // Write property tree to the stream
    exportModule.writeRawDataFile();
}

// --------------------------------------------------------------------
//  Reads a vox scene file from the stream
// --------------------------------------------------------------------
Scene RawVolumeFile::importer(ResourceIStream & source, OptionSet const& options)
{
    // Parse XML format input file into boost::property_tree
    filescope::RawImporter importModule(source, options, m_handle);

    // Read property tree and load scene
    return importModule.readRawDataFile();
}

} // namespace vox