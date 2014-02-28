/* ===========================================================================

    Project: PVM Volume Import Module
    
	Description: Defines a VoxScene import module for .pvm format volumes

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
#ifndef PVMI_PVM_VOLUME_FILE_H
#define PVMI_PVM_VOLUME_FILE_H

// Include Dependencies
#include "Plugins/PvmVolumeImporter/Common.h"
#include "VoxLib/Core/Format.h"
#include "VoxLib/Error/Error.h"
#include "VoxLib/IO/Resource.h"
#include "VoxScene/Scene.h"

// API namespace
namespace vox 
{

/**
 * PVM volume file import / export module
 *
 * This module is compatible with the abstract scene import/export interface.
 */
class PVMI_EXPORT PvmVolumeFile : public SceneExporter, public SceneImporter
{
public:
    PvmVolumeFile(std::shared_ptr<void> handle) : m_handle(handle) { }

	/** 
     * @brief Vox Scene File Exporter
     *
     * \b{Valid Options}
     *
     * \b{Required Options}
     */
	virtual void exporter(ResourceOStream & source, OptionSet const& options, Scene const& scene);

	/** 
     * @brief Vox Scene File Importer 
     *
     * \b{Valid Options}
     *
     * \b{Required Options}
     */
	virtual Scene importer(ResourceIStream & source, OptionSet const& options);
    
private:
    std::shared_ptr<void> m_handle; ///< Plugin handle to track this DLL's usage
};

}

// End definition
#endif // PVMI_PVM_VOLUME_FILE_H