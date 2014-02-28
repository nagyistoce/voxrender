/* ===========================================================================

    Project: Standard Image ExIm Module
    
	Description: Defines an image import module for common LDR image formats

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
#include "JpegImg.h"

// Include Dependencies
#include "VoxLib/Core/Common.h"
#include "VoxLib/Core/Functors.h"
#include "VoxLib/Core/Logging.h"

// Image Libraries
#include "jpeglib.h"

// API namespace
namespace vox
{

// File scope namespace
namespace
{
    namespace filescope
    {
        static size_t const BUF_SIZE = 512;

        // Export module implementation
        class ImgExporter
        {
        public:
            class DestinationManager
            {
            public:
                DestinationManager(ImgExporter * _parent) : parent(_parent) 
                {
                    mgr.empty_output_buffer = ImgExporter::emptyBuffer;
                    mgr.init_destination    = ImgExporter::initDestination;
                    mgr.term_destination    = ImgExporter::termDestination;
                }

                jpeg_destination_mgr mgr;
                ImgExporter * parent;
            };

            // --------------------------------------------------------------------
            //  Parse the scene data into a boost::property_tree
            // --------------------------------------------------------------------
            ImgExporter(ResourceOStream & sink, OptionSet const& options, Bitmap const& image, std::shared_ptr<void> handle) :
                m_sink(sink), m_options(options), m_image(image), m_handle(handle)
            {
                // Compose the resource identifier for log warning entries
                std::string filename = sink.identifier().extractFileName();
                m_displayName = filename.empty() ? "UNKNOWN" : format("\"%1%\"", filename);
            }

            // --------------------------------------------------------------------
            //  Write the boost::property_tree as an XML file to the stream
            // --------------------------------------------------------------------
            void writeImageFile()
            {
                // Setup the compression options
                struct jpeg_compress_struct cinfo;
                struct jpeg_error_mgr jerr;

                cinfo.err = jpeg_std_error(&jerr);
                jpeg_create_compress(&cinfo);

                cinfo.image_width = m_image.width();
                cinfo.image_height = m_image.height();

                bool swapRgb = false;
                unsigned int stripAlpha = 0;
                switch (m_image.type())
                {
                case Bitmap::Format_RGB:
                    cinfo.input_components = 3;
                    cinfo.in_color_space = JCS_RGB;
                    swapRgb = true;
                    break;
                case Bitmap::Format_RGBA:
                    VOX_LOG_WARNING(Error_Consistency, VOX_SIMG_LOG_CATEGORY, 
                        format("JPEG formatting will strip alpha channel <%1%>", m_displayName));
                case Bitmap::Format_RGBX: // "Alpha" channel information is junk anyway, no msg
                    cinfo.input_components = 3;
                    cinfo.in_color_space = JCS_RGB;
                    stripAlpha = 4u;
                    swapRgb = true;
                    break;
                case Bitmap::Format_Gray:
                    cinfo.input_components = 1;
                    cinfo.in_color_space = JCS_GRAYSCALE;
                    if (m_image.depth() != 8) throw Error(__FILE__, __LINE__, VOX_SIMG_LOG_CATEGORY, 
                        format("JPEG supports only 8 bit grayscale <%1%>", m_displayName), Error_NotAllowed);
                    break;
                case Bitmap::Format_GrayAlpha:
                    cinfo.input_components = 1;
                    cinfo.in_color_space = JCS_GRAYSCALE;
                    VOX_LOG_WARNING(Error_Consistency, VOX_SIMG_LOG_CATEGORY, 
                        format("JPEG formatting will strip alpha channel <%1%>", m_displayName));
                    stripAlpha = 2;
                default:
                    throw Error(__FILE__, __LINE__, VOX_SIMG_LOG_CATEGORY, 
                        format("Unsupported image format: %1%", m_image.type()), 
                        Error_NotAllowed);
                }

                jpeg_set_defaults(&cinfo);
                jpeg_set_quality(&cinfo, m_options.lookup("Quality", 100), true);

                // Setup the destination manager
                DestinationManager manager(this);
                cinfo.dest = (jpeg_destination_mgr*)&manager;

                // Perform the actual JPEG encoding
                jpeg_start_compress(&cinfo, true);
                auto data   = (char*)m_image.data();
                auto stride = m_image.stride();

                // Check if we can write the image data without any transforms
                if (!stripAlpha && !swapRgb)
                {
                    while (cinfo.next_scanline < cinfo.image_height) {
                        JSAMPROW rowPointer = (JSAMPROW)(data + cinfo.next_scanline*stride);
                        jpeg_write_scanlines(&cinfo, &rowPointer, 1);
                    }
                }
                else // We need 1 or more data transforms
                {
                    auto stride    = m_image.width()*m_image.depth()/8*(stripAlpha ? m_image.channels()-1 : 0);
                    auto rElemSize = m_image.depth()/8 * m_image.channels();
                    auto wElemSize = stripAlpha ? (stripAlpha - 1) : rElemSize;

                    std::unique_ptr<UInt8[]> buf(new UInt8[stride]);
                    JSAMPROW rowPointer = (JSAMPROW)(buf.get());
                    while (cinfo.next_scanline < cinfo.image_height) {
                        auto writePtr = (char*)buf.get();
                        auto readPtr  = (char*)m_image.data() + m_image.stride() * cinfo.next_scanline;
                        for (size_t i = 0; i < m_image.width(); i++)
                        {
                            memcpy(writePtr, readPtr, wElemSize);
                            if (swapRgb) 
                            {
                                UInt8 temp = writePtr[0];
                                writePtr[0] = writePtr[2];
                                writePtr[2] = temp;
                            }
                            writePtr += wElemSize;
                            readPtr  += rElemSize;
                        }
                        jpeg_write_scanlines(&cinfo, &rowPointer, 1);
                    }

                }

                jpeg_finish_compress(&cinfo);

                jpeg_destroy_compress(&cinfo);
            }

        private:
            static void initDestination(j_compress_ptr cinfo)
            {
                DestinationManager * mgr = (DestinationManager*)cinfo->dest;
                mgr->mgr.next_output_byte = mgr->parent->m_buffer;
                mgr->mgr.free_in_buffer = BUF_SIZE;
            }

            static boolean emptyBuffer(j_compress_ptr cinfo)
            {
                DestinationManager * mgr = (DestinationManager*)cinfo->dest;
                mgr->parent->m_sink.write((char*)mgr->parent->m_buffer, BUF_SIZE);
                mgr->mgr.next_output_byte = mgr->parent->m_buffer;
                mgr->mgr.free_in_buffer = BUF_SIZE;
                return true;
            }

            static void termDestination(j_compress_ptr cinfo)
            {
                DestinationManager * mgr = (DestinationManager*)cinfo->dest;
                mgr->parent->m_sink.write((char*)mgr->parent->m_buffer, BUF_SIZE - mgr->mgr.free_in_buffer);
            }

        private:
            static int const versionMajor = 0; ///< Scene version major [potentially breaking]
            static int const versionMinor = 0; ///< Scene version minor [usually non-breaking]

            UInt8 m_buffer[BUF_SIZE]; // Storage buffer :TODO: write directly to streambuf?

            std::shared_ptr<void> m_handle; ///< Plugin handle

            ResourceOStream & m_sink;        ///< Resource stream
            OptionSet const&  m_options;     ///< Import options
            Bitmap const&   m_image;       ///< image data
            std::string       m_displayName; ///< Warning identifier
        };

        // Import module implementation
        class ImgImporter
        {
        private:
            class SourceManager
            {
            public:
                SourceManager(ImgImporter * _parent) : parent(_parent) 
                {
                    mgr.fill_input_buffer   = ImgImporter::fillInputBuffer;
                    mgr.init_source         = ImgImporter::initSource;
                    mgr.resync_to_restart   = jpeg_resync_to_restart;
                    mgr.skip_input_data     = ImgImporter::skipInputData;
                    mgr.term_source         = ImgImporter::termSource;
                }

                jpeg_source_mgr mgr;
                ImgImporter * parent;
            };

        public:
            // --------------------------------------------------------------------
            //  Parse the resource data from an XML format into a property tree
            // --------------------------------------------------------------------
            ImgImporter(ResourceIStream & source, OptionSet const& options, std::shared_ptr<void> & handle) : 
              m_options(options), m_source(source), m_handle(handle)
            {
                // Compose the resource identifier for log warning entries
                std::string filename = source.identifier().extractFileName();
                m_displayName = filename.empty() ? "UNKNOWN" : format("\"%1%\"", filename);
            }
            
            // --------------------------------------------------------------------
            //  Read in the PNG image data using libPNG
            // --------------------------------------------------------------------
            Bitmap readImageFile()
            {
                // Setup the compression options
                struct jpeg_decompress_struct cinfo;
                struct jpeg_error_mgr jerr;

                cinfo.err = jpeg_std_error(&jerr);
                jpeg_create_decompress(&cinfo);

                SourceManager mgr(this);
                cinfo.src = (jpeg_source_mgr*)&mgr;

                jpeg_read_header(&cinfo, true);
                size_t stride = cinfo.output_width * cinfo.output_components;
                size_t bytes  = cinfo.output_height * stride;
                std::shared_ptr<UInt8> buffer(new UInt8[bytes], arrayDeleter);

                auto ptr = buffer.get();
                while (cinfo.output_scanline < cinfo.output_height)
                {
                    jpeg_read_scanlines(&cinfo, (JSAMPARRAY)&ptr, 1);
                    ptr += stride;
                }

                jpeg_finish_decompress(&cinfo);

                jpeg_destroy_decompress(&cinfo);

                Bitmap::Format type;
                switch (cinfo.out_color_space)
                {
                case JCS_RGB:
                    type = Bitmap::Format_RGB;
                    break;
                case JCS_GRAYSCALE:
                    type = Bitmap::Format_Gray;
                    break;
                default: 
                    type = Bitmap::Format_Unknown;
                    break;
                }

                return Bitmap(type, cinfo.output_width, cinfo.output_height, 8, stride, buffer);
            }
            
        private:
            static void initSource(j_decompress_ptr cinfo)
            {
                SourceManager * mgr = (SourceManager*)cinfo->src;
                mgr->mgr.bytes_in_buffer = 0;
            }

            static boolean fillInputBuffer(j_decompress_ptr cinfo)
            {
                SourceManager * mgr = (SourceManager*)cinfo->src;
                mgr->parent->m_source.read((char*)mgr->parent->m_buffer, BUF_SIZE);
                mgr->mgr.bytes_in_buffer = mgr->parent->m_source.gcount();
                mgr->mgr.next_input_byte = mgr->parent->m_buffer;

                return true;
            }
            
            static void skipInputData(j_decompress_ptr cinfo, long num_bytes)
            {
                SourceManager * mgr = (SourceManager*)cinfo->src;
                auto left = mgr->mgr.bytes_in_buffer;
                if (left > num_bytes)
                {
                    mgr->mgr.bytes_in_buffer -= num_bytes;
                    mgr->mgr.next_input_byte += num_bytes;
                }
                else
                {
                    mgr->mgr.bytes_in_buffer = 0;
                    mgr->parent->m_source.ignore(num_bytes-left);
                }
            }
            
            static void termSource(j_decompress_ptr cinfo) { }

        private:
            static int const versionMajor = 0; ///< Version major
            static int const versionMinor = 0; ///< Version minor 
            
            UInt8 m_buffer[BUF_SIZE]; // Storage buffer :TODO: write directly to streambuf?

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
void JpegImg::exporter(ResourceOStream & sink, OptionSet const& options, Bitmap const& image)
{
    // Parse scenefile object into boost::property_tree
    filescope::ImgExporter exportModule(sink, options, image, m_handle);

    // Write property tree to the stream
    exportModule.writeImageFile();
}

// --------------------------------------------------------------------
//  Reads a vox scene file from the stream
// --------------------------------------------------------------------
Bitmap JpegImg::importer(ResourceIStream & source, OptionSet const& options)
{
    // Parse XML format input file into boost::property_tree
    filescope::ImgImporter importModule(source, options, m_handle);

    // Read property tree and load scene
    return importModule.readImageFile();
}

} // namespace vox