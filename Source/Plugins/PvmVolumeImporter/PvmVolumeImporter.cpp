/* ===========================================================================

	Project: VoxRender - PVM Volume File
    
	Description: Pvm volume file import/export module

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

// Based on DDS/PVM tools in "Versatile Volume Viewer"
// (c) by Stefan Roettger, licensed under GPL 2+

// Include Header
#include "PvmVolumeImporter.h"

// Include Dependencies
#include "VoxLib/Core/Common.h"
#include "VoxLib/Scene/Volume.h"
#include "VoxLib/Scene/Scene.h"
#include "VoxLib/Core/Functors.h"
#include "VoxLib/Core/Logging.h"

#include <boost/detail/endian.hpp>

// API namespace
namespace vox
{

// File scope namespace
namespace
{
    namespace filescope
    {
        size_t const INTERLEAVE_BLOCK_SIZE = 1 << 24;

        String const DDS_TYPES[] = { "DDS v3e", "DDS v3d", "" };
        String const RAW_TYPES[] = { "PVM", "PVM2", "PVM3", "" };

        size_t const DDS_V3E = 0;
        size_t const DDS_V3D = 1;

        size_t const PVM_1 = 0;
        size_t const PVM_2 = 1;
        size_t const PVM_3 = 2;

        // Helper class for DDS file import
        class MemoryIStream: virtual public std::istream
        {
            public:
                MemoryIStream(const UInt8* data, size_t bytes) :
                    std::istream(&m_buffer),
                    m_buffer((char*)data, bytes)
                    {
                        rdbuf(&m_buffer);
                    }

            private:
                class MemoryBuffer: public std::basic_streambuf<char>
                {
                public:
                    MemoryBuffer(char* data, size_t bytes)
                    {
                        setg(data, data, data + bytes);
                    }
                };

            MemoryBuffer m_buffer;
        };

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

        // --------------------------------------------------------------------
        //  Interleaves a data block, used to group high order bytes of 
        //  multi-byte volumes for better compression
        // --------------------------------------------------------------------
        std::shared_ptr<UInt8> interleave(unsigned char * data, size_t bytes, size_t dist, size_t blockSize)
        {
            auto out = makeSharedArray(bytes);

            if (dist == 1) // No interleaving
            {
                memcpy(out.get(), data, bytes);
            }
            else if (!blockSize || bytes < blockSize) // Entire volume was interleaved/deinterleaved
            {
                auto * writePtr = out.get();
                auto * readPtr  = data;
                for (size_t i = 0; i < dist; i++)
                for (size_t j = i; j < bytes; j += dist)
                    writePtr[j] = *readPtr++;
            }
            else // Chunked before interleave/deinterleave
            {
                auto chunks = bytes / blockSize / dist;
                for (size_t chunk = 0; chunk < chunks; chunk++)
                {
                    auto offset     = chunk * blockSize * dist;
                    auto * writePtr = out.get() + offset;
                    auto * readPtr  = data + offset;
                    for (size_t i = 0; i < dist; i++)
                    for (size_t j = i; j < dist*blockSize; j += dist) 
                        writePtr[j] = *readPtr++;
                    writePtr += dist * blockSize;
                }

                auto offset = chunks * blockSize * dist;
                auto * writePtr = out.get() + offset;
                auto * readPtr  = data + offset;
                auto remains  = bytes - offset * blockSize * dist;
                for (size_t i = 0; i < dist; i++)
                for (size_t j = i; j < remains; j += dist)
                    writePtr[j] = *readPtr++;
            }

            return out;
        }

        // Export module implementation
        class PvmExporter
        {
        public:
            // --------------------------------------------------------------------
            //  Parse the scene data into a boost::property_tree
            // --------------------------------------------------------------------
            PvmExporter(ResourceOStream & sink, OptionSet const& options, Scene const& scene) :
                m_sink(sink), m_options(options), m_scene(scene)
            {
                // Compose the resource identifier for log warning entries
                std::string filename = sink.identifier().extractFileName();
                m_displayName = filename.empty() ? "UNKNOWN" : format("\"%1%\"", filename);
            }

            // --------------------------------------------------------------------
            //  Write the boost::property_tree as an XML file to the stream
            // --------------------------------------------------------------------
            void writePvmDataFile()
            {
                if (!m_scene.volume) throw Error(__FILE__, __LINE__, PVMI_LOG_CATEGORY, 
                    "Missing volume data from scene file", Error_MissingData);

                if (!m_scene.volume->extent()[3] == 1) throw Error(__FILE__, __LINE__, 
                    PVMI_LOG_CATEGORY, "PVM file format does not support 4D volume data", 
                    Error_NotAllowed);

                String type = m_options.lookup("format", RAW_TYPES[PVM_2]);
                
                // RAW format file
                for (size_t i = 0; RAW_TYPES[i] != ""; i++)
                    if (type == RAW_TYPES[i]) return writeRawDataFile(type);
                
                // DDS format file
                for (size_t i = 0; DDS_TYPES[i] != ""; i++)
                    if (type == DDS_TYPES[i]) return writeDdsDataFile(type);

                throw Error(__FILE__, __LINE__, PVMI_LOG_CATEGORY, 
                    format("Invalid data encoding format: \"%1%\"", type), 
                    Error_MissingData);
            }

        private:
            // --------------------------------------------------------------------
            //  Writes a raw pvm format volume file to the sink
            // --------------------------------------------------------------------
            void writeRawDataFile(String const& type)
            {
                auto spacing = m_scene.volume->spacing();
                auto data    = m_scene.volume->data();
                auto size    = m_scene.volume->extent();
                auto bytes   = m_scene.volume->typeToSize(m_scene.volume->type());

                if (size[3] != 1) throw Error(__FILE__, __LINE__, PVMI_LOG_CATEGORY,
                    "PVM format does not support 4D volume data", Error_NotAllowed);

                m_sink << RAW_TYPES[PVM_2] << "\n"
                       << size[0] << size[1] << size[2] << "\n";
                if (type != RAW_TYPES[PVM_2])
                    m_sink << spacing[0] << spacing[1] << spacing[2] << "\n";
                m_sink << bytes << "\n";

                m_sink.write((char*)data, size.fold(vox::mul));
            }
            
            // --------------------------------------------------------------------
            //  Writes a differential data stream encoded raw file
            // --------------------------------------------------------------------
            void writeDdsDataFile(String const& type)
            {
                throw Error(__FILE__, __LINE__, PVMI_LOG_CATEGORY, 
                    "DDS write not implemented", Error_NotImplemented);
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
        class PvmImporter
        {
        public:
            // --------------------------------------------------------------------
            //  Parse the resource data from an XML format into a property tree
            // --------------------------------------------------------------------
            PvmImporter(ResourceIStream & source, OptionSet const& options, std::shared_ptr<void> & handle) : 
                m_options(options), m_source(source), m_handle(handle), m_rem(0), m_buffer(0)
            {
                // Compose the resource identifier for log warning entries
                std::string filename = source.identifier().extractFileName();
                m_displayName = filename.empty() ? "UNKNOWN" : format("\"%1%\"", filename);
            }
            
            // --------------------------------------------------------------------
            //  Null operation, there is nothing to parse here
            // --------------------------------------------------------------------
            Scene readPvmDataFile()
            {
                return readPvmHelper(m_source);
            }
            
        private:
            // --------------------------------------------------------------------
            //  Reads a PVM format file
            // --------------------------------------------------------------------
            Scene readPvmHelper(std::istream & stream)
            {
                // Verify the 'PVM' file header
                String header;
                std::getline(stream, header);

                // RAW format file
                for (size_t i = 0; RAW_TYPES[i] != ""; i++)
                    if (header == RAW_TYPES[i]) return readRawDataFile(header, stream);
                
                // DDS format file
                for (size_t i = 0; DDS_TYPES[i] != ""; i++)
                    if (header == DDS_TYPES[i]) return readDdsDataFile(header, stream);

                // Unrecognized header
                throw Error(__FILE__, __LINE__, PVMI_LOG_CATEGORY, 
                    format("Invalid PVM file header in \"%1%\": \"%2%\"", m_displayName, header));
            }

            // --------------------------------------------------------------------
            //  Reads a differential data stream format PVM file
            // --------------------------------------------------------------------
            Scene readDdsDataFile(String const& header, std::istream & stream) 
            { 
                static size_t CHUNK_SIZE_BITS = 7;
                static size_t OFFSET_VAL_BITS = 3;
                static size_t BLOCK_SIZE = 2097408; // 128^3 8bit volume size + head fluff
                
                // Clear bit buffer
                m_active = &stream;
                m_rem = 0;

                // Decode the differential data stream
                unsigned int interDist = readBits(2) + 1;  // Interleave distance for data
                unsigned int stripLen  = readBits(16) + 1; // Length of strip between differential samples
                int value = 0;
                
                std::vector<unsigned char> data;
                data.reserve(BLOCK_SIZE);
                unsigned char * ptr = nullptr;

                while (auto count = readBits(CHUNK_SIZE_BITS))
                {
                    auto bits = readBits(OFFSET_VAL_BITS);
                    if (bits) ++bits;
                    Int32 offset = (1 << bits) / 2;

                    for (unsigned int i = 0; i < count; i++)
                    {
                        value += readBits(bits) - offset; 
                        if (data.size() > stripLen) 
                            value += *(ptr - stripLen + 1) - *(ptr - stripLen); 
                        value %= 256;

                        data.push_back(value);
                        //if (data.size() == data.capacity()) data.reserve(data.size() + data.capacity());
                        ptr = &data.back();
                    }
                }

                // Deinterleave the data
                auto blockSize = (DDS_TYPES[DDS_V3E] == header) ? INTERLEAVE_BLOCK_SIZE : 0;
                auto buffer = interleave(&data[0], data.size(), interDist, blockSize);

                // Read the resulting RAW format PVM file from memory
                return readPvmHelper(MemoryIStream(buffer.get(), data.size()));
            }
            
            // --------------------------------------------------------------------
            //  Reads a RAW format PVM file
            // --------------------------------------------------------------------
            Scene readRawDataFile(String const& header, std::istream & stream) 
            {
                // Extract the PVM header information
                Vector3u extent;
                Vector3f spacing;
                size_t   bytes;

                stream >> extent;
                if (header != RAW_TYPES[PVM_1])
                    stream >> spacing;
                stream >> bytes;

                // Deduce the volume type from the bytes per voxel count
                Volume::Type type;
                switch (bytes)
                {
                case 1: type = Volume::Type_UInt8; break;
                case 2: type = Volume::Type_UInt16; break;
                default:
                    throw Error(__FILE__, __LINE__, PVMI_LOG_CATEGORY, 
                        format("Unsupported voxel size: %1% bytes", bytes));
                }

                // Read the raw volume data 
                size_t voxels = extent.fold(&mul);
                size_t total  = voxels*bytes;
                auto data = makeSharedArray(total);
                stream.read((char*)data.get(), total);

                // Convert to the native byte ordering
                #ifdef BOOST_LITTLE_ENDIAN
                    if (type == Volume::Type_UInt16)
                    {
                        UInt16 * ptr = (UInt16*)data.get();
                        for (size_t i = 0; i < voxels; i++)
                            ptr[i] = VOX_SWAP16(ptr[i]);
                    }
                #endif

                // Construct the volume object for return
                auto offset  = m_options.lookup("Offset", Vector3f(0.0f, 0.0f, 0.0f));
                Scene scene;
                scene.volume = Volume::create(data, Vector4u(extent[0], extent[1], extent[2], 1), 
                    Vector4f(spacing[0], spacing[1], spacing[2], 1), offset, type);

                return scene; 
            }
            
            // --------------------------------------------------------------------
            //  Reads a number of bits from the active stream
            // --------------------------------------------------------------------
            UInt32 readBits(size_t bits)
            {
                if (bits == 0) return 0;

                if (bits > 32) throw Error(__FILE__, __LINE__, PVMI_LOG_CATEGORY,
                    "File is corrupt or invalid (offset bits > 32)", Error_BadFormat);

                if (bits <= m_rem)
                {
                    m_rem -= bits;
                    UInt32 mask  = ((1 << bits) - 1);
                    UInt32 value = (m_buffer >> m_rem) & mask;
                    return value; 
                }

                UInt32 left = bits - m_rem;
                UInt32 value = (readBits(m_rem) << left);
                bufferBits();
                value |= readBits(left);

                return value;
            }
            
            // --------------------------------------------------------------------
            //  Buffers additional bytes from the active stream
            // --------------------------------------------------------------------
            void bufferBits()
            {
                if (m_rem != 0) return;
                
                m_active->read((char*)&m_buffer, sizeof(m_buffer));
                m_rem = sizeof(m_buffer)*8;

                #ifdef BOOST_LITTLE_ENDIAN
                    m_buffer = VOX_SWAP32(m_buffer);
                #endif
            }

            static int const versionMajor = 0; ///< Version major
            static int const versionMinor = 0; ///< Version minor 
            
            std::istream * m_active; ///< Active bit stream
            UInt32 m_buffer;         ///< Buffer for bit-level reads
            size_t m_rem;            ///< Valid bytes remaining in buffer

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
void PvmVolumeFile::exporter(ResourceOStream & sink, OptionSet const& options, Scene const& scene)
{
    // Parse scenefile object into boost::property_tree
    filescope::PvmExporter exportModule(sink, options, scene);

    // Write property tree to the stream
    exportModule.writePvmDataFile();
}

// --------------------------------------------------------------------
//  Reads a vox scene file from the stream
// --------------------------------------------------------------------
Scene PvmVolumeFile::importer(ResourceIStream & source, OptionSet const& options)
{
    // Parse XML format input file into boost::property_tree
    filescope::PvmImporter importModule(source, options, m_handle);

    // Read property tree and load scene
    return importModule.readPvmDataFile();
}

} // namespace vox