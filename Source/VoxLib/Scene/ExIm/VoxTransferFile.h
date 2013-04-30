/* ===========================================================================

	Project: VoxRender - Transfer File
    
	Description: Vox transfer file import/export module

    Copyright (C) 2012 Lucas Sherman

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
#ifndef VOX_TRANSFER_FILE_H
#define VOX_TRANSFER_FILE_H

// Include Dependencies
#include "VoxLib/Core/Common.h"
#include "VoxLib/Core/Format.h"
#include "VoxLib/Error/Error.h"
#include "VoxLib/IO/Resource.h"
#include "VoxLib/Scene/Scene.h"

// API namespace
namespace vox 
{

/**
 * Vox transfer file import / export module
 *
 * This module is compatible with the abstract scene import/export interface.
 */
class VOX_EXPORT VoxTransferFile
{
public:
	/** 
     * @brief Vox Scene File Exporter
     *
     * \b{Valid Options}
     *
     * \b{Required Options}
     *
     * @throws
     *  ::vox::Error
     *  ::std::exception
     */
	static void exporter(ResourceOStream & data, OptionSet const& options, Scene const& scene);

	/** 
     * @brief Vox Scene File Importer 
     *
     * \b{Valid Options}
     *
     * \b{Required Options}
     *
     * @throws
     *  ::vox::Error
     *  ::std::exception
     */
	static Scene importer(ResourceIStream & data, OptionSet const& options);

private:
    VoxTransferFile( ) { }
};

}

// End definition
#endif // VOX_TRANSFER_FILE_H