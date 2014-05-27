/* ===========================================================================

	Project: VoxRender - Image
    
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

// Begin definition
#ifndef VOX_RAW_IMAGE_H
#define VOX_RAW_IMAGE_H

// Include Dependencies
#include "VoxLib/Core/CudaCommon.h"
#include "VoxLib/Core/Types.h"
#include "VoxLib/IO/Resource.h"

// API namespace
namespace vox
{
    class ImageImporter;
    class ImageExporter;

	/** 
	 * @brief Image Class
	 */
	class VOX_EXPORT Bitmap
	{
    public:
        /** 
         * Defines standard formats for images 
         *
         * Images are always assigned a format. For unknown formats, Format_Unknown 
         * will be assigned. For user defined formats, ensure that the format value 
         * is outside the range [Format_Begin, Format_End).
         */
        enum Format
        {
            Format_Begin,
            Format_RGB = Format_Begin,
            Format_RGBA,
            Format_RGBX,
            Format_Gray,
            Format_GrayAlpha,
            Format_YUV420p,
            Format_Unknown,
            Format_End
        };

	public:
		/**
		 * @brief Overload for imprt
         *
         * This function overloads the imprt function and automatically provides the 
         * identifier as the resource identifer and matchname and the matchname for 
         * selecting an Importer.
		 */
	    inline static Bitmap imprt(ResourceId const& identifier, OptionSet const& options = OptionSet())
        {
            return imprt(ResourceIStream(identifier), options);
        }

		/**
		 * @brief Imports a image using the internal image file loading mechanism.
		 * 
         * This function can be used as an abstract interface for importing image
         * files. The data parameter is transparently passed to the selected 
         * importer. Any exceptions thrown by an importer will be propogated out.
         *
         * @sa 
         *  ::ImageExporter
         *
		 * @param data       [in] The data stream with the imported content
         * @param matchname  [in] The match string for the importer selection
         * @param options    [in] Optional string containing 'name: value' options 
         *
         * @returns The loaded image object
         *
         * @throws
         *  ::Error No import module is defined which accepts the matchname
		 */
	    static Bitmap imprt(ResourceIStream & data, OptionSet const& options = OptionSet()); 

        
        /** Overload of imprt which requires specification of format */
	    static Bitmap imprt(std::istream & data, String const& extension, 
                            OptionSet const& options = OptionSet());

		/**
		 * @brief Overload for exprt
         *
         * This function overloads the imprt function and automatically provides the 
         * identifier as the resource identifer
		 */
	    inline void exprt(ResourceId const& identifier, OptionSet const& options = OptionSet()) const
        {
            return exprt(ResourceOStream(identifier), options);
        }

		/**
		 * @brief Exports a image using the internal image file export mechanism.
		 * 
         * This function can be used as an abstract interface for exporting image
         * files. The data parameter is transparently passed to the selected 
         * exporter. Any exceptions thrown by an exporter will be propogated out.
         *
         * @sa 
         *  ::ImageExporter
         *
		 * @param data       [out] The data stream for the exported content
         * @param matchname  [in]  The match string for the exporter selection
         *
         * @throws
         *  ::Error No export module is defined which accepts the matchname
		 */
		void exprt(ResourceOStream & data, OptionSet const&  options = OptionSet()) const; 

        /** Overload of exprt which requires specification of format */
	    void exprt(std::ostream & data, String const& extension, 
                   OptionSet const& options = OptionSet()) const;

		/**`
		 * Registers a new image importer with the specified extension. If an importer is already 
         * specified which has a conflicting extension, it will be overridden.
		 * 
		 * @param importer [in] The new image importer to be registered
		 * @param matcher  [in] The regular expression for matching
		 */
        static void registerImportModule(String const& extension, std::shared_ptr<ImageImporter> importer);

		/**
		 * Registers a new image exporter with the specified extension.
         * If an exporter is already specified which has a conflicting extension 
         * the new exporter will take precedence.
		 * 
		 * @param loader  [in] The new image exporter to be registered
		 * @param matcher [in] The regular expression for matching
		 */
        static void registerExportModule(String const& extension, std::shared_ptr<ImageExporter> exporter);

        /** Removes an image import module */
        static void removeImportModule(std::shared_ptr<ImageImporter> importer, String const& extension = "");

        /** Removes an image export module */
        static void removeExportModule(std::shared_ptr<ImageExporter> exporter, String const& extension = "");
        
        /** Removes an image import module */
        static void removeImportModule(String const& extension);

        /** Removes an image export module */
        static void removeExportModule(String const& extension);
        
        /** Registers a new image conversion module */
        static void registerConvertModule(int srcFormat, int dstFormat, std::shared_ptr<void> converter);

        /** Removes an image conversion module */
        static void removeConvertModule(int srcFormat, int dstFormat);

    public:
        /** Default constructor */
        Bitmap() { reset(Format_Unknown, 0, 0, 0, 0, 0, 0); }

		/** Initializes an image object for a known image format */
		Bitmap(Format type, unsigned int width = 0, unsigned int height = 0, unsigned int bitDepth = 0, 
            unsigned int stride = 0, std::shared_ptr<void> data = nullptr);

        /** Initializes an image object for an unknown image format */
        Bitmap(unsigned int width, unsigned int height, unsigned int bitDepth, unsigned int nChannels, 
            unsigned int stride = 0, std::shared_ptr<void> data = nullptr)
        {
            if (data) reset(Format_Unknown, width, height, bitDepth, nChannels, 1, stride, 
                            std::list<std::shared_ptr<void>>(1, data));
            else reset(Format_Unknown, width, height, bitDepth, nChannels, 1, stride);
        }
  
        /** Advanced bitmap constructor */
        Bitmap(int type, unsigned int width, unsigned int height, unsigned int bitDepth, 
            unsigned int nChannels, unsigned int nLayers, unsigned int stride, 
            std::list<std::shared_ptr<void>> layerData = std::list<std::shared_ptr<void>>())
        {
            reset(type, width, height, bitDepth, nChannels, nLayers, stride, layerData);
        }

        /** Resets the general attributes of a bitmap */
        void reset(int type, unsigned int width, unsigned int height, unsigned int bitDepth,
                   unsigned int nChannels, unsigned int nLayers, unsigned int stride,
                   std::list<std::shared_ptr<void>> layerData = std::list<std::shared_ptr<void>>());

        /** Performs conversion between bitmap formats */
        void convert(int destinationFormat, OptionSet options = OptionSet());
        
        /** Performs conversion between bitmap formats */
        Bitmap convertCopy(int destinationFormat, OptionSet options = OptionSet());

        /** 
         * Adjusts the padding of the image 
         *
         * If newStride is set to 0, the padding will be word aligned
         *
         * @param newStride The new row stride (in bytes)
         * @param copyData  If true, original image data will be retained
         */
        void pad(size_t newStride = 0, bool copyData = true);

        /** Returns a deep copy of the image */
        Bitmap copy() const;

        /** Returns the raw image data pointer */
        void const* data(unsigned int layer = 0) const;

        /** Returns the raw image data pointer */
        void * data(unsigned int layer = 0);

        /** Resizes (does not scale) the image to the specified dimensions */
        void resize(size_t width, size_t height, size_t stride = 0);

        /** Adds an additional layer to the image */
        void addLayer(std::shared_ptr<void> layer = nullptr, unsigned int index = 0);

        /** Removes a layer from the image */
        void removeLayer(unsigned int layer);

        /** Returns the format of the image */
        int type() const { return m_format; }

        /** Returns the image bit depth */
        unsigned int depth() const { return m_depth; }

        /** Returns the number of channels in the image */
        unsigned int channels() const { return m_channels; }

        /** Returns the number of layers in the image */
        unsigned int layers() const { return m_buffer.size(); }

        /** Returns the size in bytes of an image pixel */
        unsigned int elementSize() const { return m_depth*m_channels/8; }

        /** Image width accessor */
        unsigned int width() const { return m_width; }

        /** Image height accessor */
        unsigned int height() const { return m_height; }

        /** Image stride accessor */
        unsigned int stride() const { return m_stride; }

        /** Returns the size of the image content */
        unsigned int size() const { return m_stride*m_height; }

	private:
        unsigned int m_height;   ///< Image height
		unsigned int m_width;    ///< Image width
		unsigned int m_stride;   ///< Image stride
        unsigned int m_channels; ///< Number of data channels
        unsigned int m_depth;    ///< The size of an image pixel element 
        int          m_format;   ///< Image data format

        std::vector<std::shared_ptr<void>> m_buffer; ///< Image layer data
	};

	/** 
	 * Image File Importer
     *
     * This class specifies the format of a image file importer. The VoxRender
     * library offers a built in system for controlling the import of image information. 
     * Image import/export modules are registered with the Image class through a static
     * interface and associated with a regular expression. When an attempt is made to 
     * import a image file through the Image::import interface, the first ImageLoader whose
     * regular expression matches the parameter provided to load will be executed.
     *
     * This mechanism makes it easy to load image files by overloading the load function
     * for different file types or transfer mechanisms. If a image file is requested from 
     * a remote resource repository for instance, a specific importer can be setup to match 
     * the transfer protocol and perform the file transfer before executing a load function
     * again for the specific file format which was transferred.
     *
     * There are no restriction on what a image loader may throw, and any exceptions will be
     * propogated out of the abstract load interface. As the default loaders will throw exceptions
     * derived from vox::Error, it may be most usefull to enforce that all importers throw only
     * exceptions derived from std::runtime_error or std::error.
     *
     * The option set for the load operation takes the form of 'name : value' pairs stored in
     * a map of values to boost::any objects which represent the specific value type. These name
     * value pairs are intended to allow more control over import/export behavior through the 
     * abstract interface.
     *
     * @sa
     *  ::ImageExporter
	 */              
    class ImageImporter { public: virtual Bitmap importer(std::istream & data, OptionSet const& options) = 0;
                          virtual ~ImageImporter() { } };

    /**
	 * Image File Exporter
     *
     * This typedef specifies the format of a image file exporter. The VoxRender
     * library offers a built in system for controlling the export of image information. 
     * Image import/export modules are registered with the Image class through a static
     * interface and associated with a regular expression. When an attempt is made to 
     * export a image file through the Image::export interface, the first ImageExporter whose
     * extension matches the parameter provided to load will be executed.
     *
     * @sa
     *  ::ImageImporter
     */
    class ImageExporter { public: virtual void exporter(std::ostream & data, OptionSet const& options, 
                                                        Bitmap const& image) = 0; 
                          virtual ~ImageExporter() { } };
    
    /**
     * Image format conversion class
     */
    class ImageConverter 
    { 
    public: 
        virtual void convert(Bitmap const& src, Bitmap * dst, int dstFormat, OptionSet const& options) = 0; 
                          
        virtual ~ImageConverter() { } 
    };
}

// End definition
#endif // VOX_SCENE_H