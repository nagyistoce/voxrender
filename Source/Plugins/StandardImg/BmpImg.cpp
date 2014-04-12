/* ===========================================================================

    Project: Standard Image ExIm Module
    
	Description: Defines an image import module for common LDR image formats

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
#include "BmpImg.h"

// Include Dependencies
#include "VoxLib/Core/Common.h"
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
        static size_t const BUF_SIZE = 512;
        static UInt16 const BMP_HEAD = 0x4D42;

        enum CompressionMethod
        {
            BI_RGB  = 0,
            BI_RLE8 = 1,
            BI_RLE4 = 2,
            BI_BITFIELDS = 3,
            BI_JPEG = 4,
            BI_PNG = 5,
            BI_ALPHABITFIELDS = 6
        };

        #pragma pack(push, 1)
        struct BitmapCoreHeader
        {
            UInt32 headerSize;
            UInt32 imageWidth;
            UInt32 imageHeight;
            UInt16 nColorPlanes;
            UInt16 bitsPerPixel;
        };

        struct BitmapInfoHeader
        {
            UInt32 headerSize;
            UInt32 imageWidth;
            UInt32 imageHeight;
            UInt16 nColorPlanes;
            UInt16 bitsPerPixel;
            UInt32 compressionMethod;
            UInt32 imageSize;
            Int32  horizontalResolution;
            Int32  verticalResolution;
            UInt32 nColors;
            UInt32 nImportantColors;
        };

        struct BitmapFileHeader
        {
            UInt16 header;
            UInt32 size;
            UInt16 reserved1;
            UInt16 reserved2;
            UInt32 offset;
        };
        #pragma pack(pop)

        // Export module implementation
        class ImgExporter
        {
        public:
            // --------------------------------------------------------------------
            //  Parse the scene data into a boost::property_tree
            // --------------------------------------------------------------------
            ImgExporter(std::ostream & sink, OptionSet const& options, Bitmap const& image) :
                m_sink(sink), m_options(options), m_image(image)
            {
            }

            // --------------------------------------------------------------------
            //  Write the boost::property_tree as an XML file to the stream
            // --------------------------------------------------------------------
            void writeImageFile()
            {
                // Compose the bitmap file header
                BitmapFileHeader fileHeader;
                fileHeader.header = BMP_HEAD;
                fileHeader.reserved1 = 0;
                fileHeader.reserved2 = 0;

                // Compose the bitmap DIB header
                BitmapInfoHeader dibHeader;
                dibHeader.headerSize   = sizeof(dibHeader);
                dibHeader.imageHeight  = m_image.height();
                dibHeader.imageWidth   = m_image.width();
                dibHeader.bitsPerPixel = m_image.channels() * m_image.depth();
                dibHeader.nColorPlanes = 1;
                dibHeader.nColors      = 0;
                dibHeader.compressionMethod    = BI_RGB;
                dibHeader.horizontalResolution = 0;
                dibHeader.verticalResolution   = 0;
                dibHeader.nImportantColors     = 0;

                // Compose the information for the pixel data layout
                size_t rowSize = ((dibHeader.bitsPerPixel * (size_t)dibHeader.imageWidth + 31) / 32) * 4;
                auto stride  = m_image.stride();
                auto readPtr = (UInt8*)m_image.data() + stride * (m_image.height()-1);
                auto wBytes  = m_image.width() * m_image.elementSize();
                fileHeader.offset = sizeof(fileHeader) + sizeof(dibHeader);
                fileHeader.size   = fileHeader.offset + rowSize/8 * dibHeader.imageHeight;
                // :TODO: Detect Big endian system

                // Write the bitmap file data to to output sink
                m_sink.write((char*)&fileHeader, sizeof(fileHeader));
                m_sink.write((char*)&dibHeader, sizeof(dibHeader));
                for (size_t row = 0; row < dibHeader.imageHeight; row++)
                {
                    m_sink.write((char*)readPtr, wBytes);
                    for (size_t i = 0; i < stride - wBytes; i++) 
                        m_sink.write("0", 1);
                    readPtr -= stride;
                }
            }

        private:
            static int const versionMajor = 0; ///< Scene version major [potentially breaking]
            static int const versionMinor = 0; ///< Scene version minor [usually non-breaking]

            UInt8 m_buffer[BUF_SIZE]; // Storage buffer :TODO: write directly to streambuf?

            std::ostream & m_sink;        ///< Resource stream
            OptionSet const&  m_options;     ///< Import options
            Bitmap const&   m_image;       ///< image data
            std::string       m_displayName; ///< Warning identifier
        };

        // Import module implementation
        class ImgImporter
        {
        public:
            // --------------------------------------------------------------------
            //  Parse the resource data from an XML format into a property tree
            // --------------------------------------------------------------------
            ImgImporter(std::istream & source, OptionSet const& options, std::shared_ptr<void> & handle) : 
              m_options(options), m_source(source), m_handle(handle)
            {
            }
            
            // --------------------------------------------------------------------
            //  Read in the PNG image data using libPNG
            // --------------------------------------------------------------------
            Bitmap readImageFile()
            {
                size_t bytesRead = 0;

                BitmapFileHeader fileHeader;
                bytesRead += read((char*)&fileHeader, sizeof(fileHeader));
                // :TODO: Big endian detection
                if (fileHeader.header != BMP_HEAD) corrupt();

                BitmapInfoHeader dibHeader;
                bytesRead += read((char*)&dibHeader, sizeof(BitmapCoreHeader));
                
                // Attempt to read additional header info to ensure there is no compression (NotSupported)
                if (dibHeader.headerSize >= sizeof(BitmapInfoHeader))
                {
                    auto bytes = sizeof(BitmapInfoHeader) - sizeof(BitmapCoreHeader);
                    bytesRead += read(((char*)&dibHeader)+sizeof(BitmapCoreHeader), bytes);

                    if (dibHeader.compressionMethod != BI_RGB) throw Error(
                        __FILE__, __LINE__, VOX_SIMG_LOG_CATEGORY, 
                        format("Bitmap reader does not support compression <%1%>", m_displayName),
                        Error_NotImplemented);
                }
                
                // Attempt to detect the type of the underlying bitmap 
                // (Bitmap doesn't support color palettes)
                // :TODO: If there is a palette, force unknown format
                auto type = Bitmap::Format_Unknown;
                switch (dibHeader.bitsPerPixel)
                {
                case 8: type = Bitmap::Format_Gray; break;
                case 24: type = Bitmap::Format_RGB; break;
                case 32: type = Bitmap::Format_RGBA; break;
                }

                // Read the raw image data from the file
                m_source.ignore(fileHeader.offset - bytesRead);
                if (!m_source) corrupt();
                size_t stride  = dibHeader.imageWidth * dibHeader.bitsPerPixel / 8; 
                size_t rowSize = ((dibHeader.imageWidth * dibHeader.bitsPerPixel + 31)/32) * 4;
                size_t bytes   = dibHeader.imageHeight * stride;
                auto data = makeSharedArray(bytes);
                auto readPtr = ((char*)data.get()) + bytes;
                for (size_t i = 0; i < dibHeader.imageHeight; i++)
                {
                    readPtr -= stride;
                    read((char*)data.get(), stride);
                }

                // Ensure the proper bit depth will be selected. ie format unknown require user specification
                auto bitDepth = (type == Bitmap::Format_Unknown) ? dibHeader.bitsPerPixel : 0;
                return Bitmap(type, dibHeader.imageWidth, dibHeader.imageHeight, 0, 1, stride, data);
            }
            
        private:
            void corrupt()
            {
                throw Error(__FILE__, __LINE__, VOX_SIMG_LOG_CATEGORY, 
                    format("Bitmap file corrupt or invalid <%1%>", m_displayName));
            }

            size_t read(char * writePtr, size_t bytes)
            {
                m_source.read(writePtr, bytes);
                auto bytesRead = m_source.gcount();
                if (bytesRead != bytes) corrupt();
                return bytesRead;
            }

        private:
            static int const versionMajor = 0; ///< Version major
            static int const versionMinor = 0; ///< Version minor 
            
            UInt8 m_buffer[BUF_SIZE]; // Storage buffer :TODO: write directly to streambuf?

            std::shared_ptr<void> & m_handle;   ///< Plugin handle to track this DLL's usage

            std::istream & m_source;       ///< Resource stream
            OptionSet const&  m_options;      ///< Import options
            std::string       m_displayName;  ///< Warning identifier
        };
    }
}

// --------------------------------------------------------------------
//  Writes a raw volume file to the stream
// --------------------------------------------------------------------
void BmpImg::exporter(std::ostream & sink, OptionSet const& options, Bitmap const& image)
{
    // Parse scenefile object into boost::property_tree
    filescope::ImgExporter exportModule(sink, options, image);

    // Write property tree to the stream
    exportModule.writeImageFile();
}

// --------------------------------------------------------------------
//  Reads a vox scene file from the stream
// --------------------------------------------------------------------
Bitmap BmpImg::importer(std::istream & source, OptionSet const& options)
{
    // Parse XML format input file into boost::property_tree
    filescope::ImgImporter importModule(source, options, m_handle);

    // Read property tree and load scene
    return importModule.readImageFile();
}

} // namespace vox