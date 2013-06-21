/* ===========================================================================

    Project: Raw Volume Import Module
    
	Description: Defines a VoxScene import module for .raw format volumes

    Copyright (C) 2012-2013 Lucas Sherman

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
#ifndef RVI_RAW_VOLUME_FILE_H
#define RVI_RAW_VOLUME_FILE_H

// Include Dependencies
#include "Plugins/RawVolumeImporter/Common.h"
#include "VoxLib/Core/Format.h"
#include "VoxLib/Error/Error.h"
#include "VoxLib/IO/Resource.h"
#include "VoxLib/Scene/Scene.h"

// API namespace
namespace vox 
{

/**
 * Raw volume file import / export module
 *
 * This module is compatible with the abstract scene import/export interface.
 */
class RVI_EXPORT RawVolumeFile : public SceneExporter, public SceneImporter
{
public:
	/** 
     * @brief Vox Scene File Exporter
     *
     * \b{Valid Options}
     *  Compression : String [default='']  | Specifies an ordered list of compression types to be applied
     *
     * \b{Required Options}
     */
	virtual void exporter(ResourceOStream & source, OptionSet const& options, Scene const& scene);

	/** 
     * @brief Vox Scene File Importer 
     *
     * \b{Valid Options}
     *  Compression : String [default='']  | Specifies an ordered list of compression types to be applied
     *
     * \b{Required Options}
     *  BytesPerVoxel : size_t   | Specifies the number of bytes per data voxel
     *  Size          : Vector4u | Specifies the extent of the 3D volume in the order [x y z t]
     *  Endianess     : String   | Specifies endianess ("little" or "big") Optional if BytesPerVoxel=8.
     */
	virtual Scene importer(ResourceIStream & source, OptionSet const& options);
};

}

// End definition
#endif // RVI_RAW_VOLUME_FILE_H