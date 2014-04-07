/* ===========================================================================

    Project: Standard Image ExIm Module
    
	Description: Defines an image import module for common LDR image formats
    
    Copyright (C) 2013-2014 Lucas Sherman

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
#include "PngImg.h"

// Include Dependencies
#include "VoxLib/Core/Common.h"
#include "VoxLib/Core/Functors.h"
#include "VoxLib/Core/Logging.h"

// Image Libraries
#include "png.h"

// API namespace
namespace vox
{

// File scope namespace
namespace
{
    namespace filescope
    {
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
                // Initialize the necessary libpng data structures
                auto pngPtr = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
                if (!pngPtr) throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY, "Call to png_create_write_struct failed.");

                auto infoPtr = png_create_info_struct(pngPtr);
                if (!infoPtr)
                {
                    png_destroy_read_struct(&pngPtr, nullptr, nullptr);
                    throw Error(__FILE__, __LINE__, VOX_SIMG_LOG_CATEGORY, "call to png_create_info_struct failed.");
                }

                png_set_write_fn(pngPtr, &m_sink, writeData, flushData);

                // Write the image info so we can begin encoding the image
                int colorType;
                switch (m_image.type())
                {
                case Bitmap::Format_RGB:       colorType = PNG_COLOR_TYPE_RGB; png_set_bgr(pngPtr); break;
                case Bitmap::Format_Gray:      colorType = PNG_COLOR_TYPE_GRAY; break;
                case Bitmap::Format_RGBA:      colorType = PNG_COLOR_TYPE_RGBA; png_set_bgr(pngPtr); break;
                case Bitmap::Format_RGBX:      colorType = PNG_COLOR_TYPE_RGBA; png_set_bgr(pngPtr); png_set_strip_alpha(pngPtr); break;
                case Bitmap::Format_GrayAlpha: colorType = PNG_COLOR_TYPE_GRAY_ALPHA; break;
                }
                png_set_IHDR(pngPtr, infoPtr, m_image.width(), m_image.height(), m_image.depth(), colorType, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
                png_write_info(pngPtr, infoPtr);

                // Write the actual image data
                auto height  = m_image.height();
                auto stride  = m_image.stride(); 
                auto data    = (char*)m_image.data();
                auto rowPtrs = new png_bytep[height];
                for (size_t i = 0; i < height; i++) 
                    rowPtrs[i] = (png_bytep)data + i*stride;
                png_write_image(pngPtr, rowPtrs);
                png_write_end(pngPtr, infoPtr);

                png_destroy_write_struct(&pngPtr, &infoPtr);
            }

        private:
            // --------------------------------------------------------------------
            //  Writes the PNG image data to the sink stream
            // --------------------------------------------------------------------
            static void writeData(png_structp pngPtr, png_bytep data, png_size_t length) 
            {
                auto strPtr = png_get_io_ptr(pngPtr);
                ((std::ostream*)strPtr)->write((char*)data, length);
            }

            // --------------------------------------------------------------------
            //  Writes the PNG image data to the sink stream
            // --------------------------------------------------------------------
            static void flushData(png_structp pngPtr) 
            {
                auto strPtr = png_get_io_ptr(pngPtr);
                ((std::ostream*)strPtr)->flush();
            }

        private:
            static int const versionMajor = 0; ///< Scene version major [potentially breaking]
            static int const versionMinor = 0; ///< Scene version minor [usually non-breaking]

            std::ostream &   m_sink;        ///< Resource stream
            OptionSet const& m_options;     ///< Import options
            Bitmap const&    m_image;       ///< image data
            std::string      m_displayName; ///< Warning identifier
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
                // Initialize the necessary libpng data structures
                auto pngPtr = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
                if (!pngPtr) throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY, "Call to png_create_read_struct failed.");

                auto infoPtr = png_create_info_struct(pngPtr);
                if (!infoPtr)
                {
                    png_destroy_read_struct(&pngPtr, nullptr, nullptr);
                    throw Error(__FILE__, __LINE__, VOX_SIMG_LOG_CATEGORY, "call to png_create_info_struct failed.");
                }

                png_set_read_fn(pngPtr, &m_source, readData);

                // Read the image info so we can initialize the image structure
                png_read_info(pngPtr, infoPtr);
                auto imgWidth   = png_get_image_width(pngPtr, infoPtr);
                auto imgHeight  = png_get_image_height(pngPtr, infoPtr);
                auto bitdepth   = png_get_bit_depth(pngPtr, infoPtr);
                auto channels   = png_get_channels(pngPtr, infoPtr);
                auto color_type = png_get_color_type(pngPtr, infoPtr);

                Bitmap::Format format;
                switch (color_type)
                {
                case PNG_COLOR_TYPE_RGB:        format = Bitmap::Format_RGB; break;
                case PNG_COLOR_TYPE_GRAY:       format = Bitmap::Format_Gray; break;
                case PNG_COLOR_TYPE_RGB_ALPHA:  format = Bitmap::Format_RGBA; break;
                case PNG_COLOR_TYPE_GRAY_ALPHA: format = Bitmap::Format_GrayAlpha; break;
                }
                Bitmap result(format, imgWidth, imgHeight, bitdepth, 0); 

                // Perform the actual image decoding
                auto stride  = result.stride(); 
                auto data    = (char*)result.data();
                auto rowPtrs = new png_bytep[imgHeight];
                for (size_t i = 0; i < imgHeight; i++) 
                    rowPtrs[i] = (png_bytep)data + i*stride;
                png_read_image(pngPtr, rowPtrs);

                png_destroy_read_struct(&pngPtr, nullptr, nullptr);

                return result;
            }
            
        private:
            // --------------------------------------------------------------------
            //  Reads the PNG image data from the source stream
            // --------------------------------------------------------------------
            static void readData(png_structp pngPtr, png_bytep data, png_size_t length) 
            {
                auto strPtr = png_get_io_ptr(pngPtr);
                ((std::istream*)strPtr)->read((char*)data, length);
            }

        private:
            static int const versionMajor = 0; ///< Version major
            static int const versionMinor = 0; ///< Version minor 
            
            std::shared_ptr<void> & m_handle;   ///< Plugin handle to track this DLL's usage

            std::istream &   m_source;       ///< Resource stream
            OptionSet const& m_options;      ///< Import options
            std::string      m_displayName;  ///< Warning identifier
        };
    }
}

// --------------------------------------------------------------------
//  Writes a raw volume file to the stream
// --------------------------------------------------------------------
void PngImg::exporter(std::ostream & sink, OptionSet const& options, Bitmap const& image)
{
    // Parse scenefile object into boost::property_tree
    filescope::ImgExporter exportModule(sink, options, image);

    // Write property tree to the stream
    exportModule.writeImageFile();
}

// --------------------------------------------------------------------
//  Reads a vox scene file from the stream
// --------------------------------------------------------------------
Bitmap PngImg::importer(std::istream & source, OptionSet const& options)
{
    // Parse XML format input file into boost::property_tree
    filescope::ImgImporter importModule(source, options, m_handle);

    // Read property tree and load scene
    return importModule.readImageFile();
}

} // namespace vox