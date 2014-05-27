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

// Begin definition
#ifndef VOX_SIMG_JPEG_IMG_H
#define VOX_SIMG_JPEG_IMG_H

// Include Dependencies
#include "Plugins/StandardImg/Common.h"
#include "VoxLib/Core/Format.h"
#include "VoxLib/Error/Error.h"
#include "VoxLib/IO/Resource.h"
#include "VoxLib/Bitmap/Bitmap.h"

// API namespace
namespace vox 
{

/**
 * Standard image file import / export module
 *
 * This module is compatible with the abstract scene import/export interface.
 */
class VOX_SIMG_EXPORT JpegImg : public ImageExporter, public ImageImporter
{
public:
    enum Container
    {
        Container_JPG,
        Container_MPO,
        Container_JPS
    };

public:
    JpegImg(std::shared_ptr<void> handle, Container container = Container_JPG) : 
        m_handle(handle), m_container(container) { }

	/** Vox Image File Exporter */
	virtual void exporter(std::ostream & source, OptionSet const& options, Bitmap const& image);

	/** Vox Image File Importer */
	virtual Bitmap importer(std::istream & source, OptionSet const& options);
    
private:
    std::shared_ptr<void> m_handle; ///< Plugin handle to track this DLL's usage

    Container m_container;
};

}

// End definition
#endif // VOX_SIMG_JPEG_IMG_H