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
	class VOX_EXPORT RawImage
	{
    public:
        enum Format
        {
            Format_Begin,
            Format_RGB = Format_Begin,
            Format_RGBA,
            Format_RGBX,
            Format_Gray,
            Format_GrayAlpha,
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
	    inline static RawImage imprt(ResourceId const& identifier, OptionSet const& options = OptionSet())
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
	    static RawImage imprt(ResourceIStream & data, 
                              OptionSet const&  options   = OptionSet(),
                              String const&     extension = String()); 

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
		void exprt(ResourceOStream & data, 
                   OptionSet const&  options   = OptionSet(), 
                   String const&     extension = String()) const; 

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

    public:
		/** Initializes an empty image structure */
		RawImage(Format type, size_t width = 0, size_t height = 0, size_t bitDepth = 0, size_t stride = 0, std::shared_ptr<void> data = nullptr);

        /** Returns the raw image data pointer */
        void * data() const { return m_buffer.get(); }

        /** Returns the raw image data pointer */
        void * data() { return m_buffer.get(); }

        /** Resizes the image to the specified dimensions */
        void resize(size_t width, size_t height)
        {
            resize(width, height, width*m_depth*m_channels);
        }

        /** Resizes the image to the specified dimensions */
        void resize(size_t width, size_t height, size_t stride)
        {
            m_width = width; m_height = height; m_stride = stride;

            m_buffer = std::shared_ptr<void>(new UInt8[stride*height]);
        }
        
        /** Returns the format of the image */
        Format type() const { return m_format; }

        /** Returns the image bit depth */
        size_t depth() const { return m_depth; }

        /** Returns the number of channels on the image */
        size_t channels() const { return m_channels; }

        /** Returns the size in bytes of an image pixel */
        size_t elementSize() const { return m_depth*m_channels; }

        /** Image width accessor */
        size_t width() const { return m_width; }

        /** Image height accessor */
        size_t height() const { return m_height; }

        /** Image stride accessor */
        size_t stride() const { return m_stride; }

        /** Returns the size of the image content */
        size_t size() const { return m_stride*m_height; }

	private:
        size_t m_height;   ///< Image height
		size_t m_width;    ///< Image width
		size_t m_stride;   ///< Image stride
        size_t m_channels; ///< Number of data channels
        Format m_format;   ///< Image data format
        size_t m_depth;    ///< The size of an image pixel element 

        std::shared_ptr<void> m_buffer;  ///< Image buffer
	};

	/** 
	 * Image File Importer
     *
     * This typedef specifies the format of a image file importer. The VoxRender
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
    class ImageImporter { public: virtual RawImage importer(ResourceIStream & data, OptionSet const& options) = 0;
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
    class ImageExporter { public: virtual void exporter(ResourceOStream & data, OptionSet const& options, 
                                                        RawImage const& image) = 0; 
                          virtual ~ImageExporter() { } };
}

// End definition
#endif // VOX_SCENE_H