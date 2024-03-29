/* ===========================================================================

    Project: Vox Scene Importer - Module definition for scene importer

    Description: A vox scene file importer module

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
#ifndef VSI_SCENE_FILE_H
#define VSI_SCENE_FILE_H

// Include Dependencies
#include "Plugins/VoxSceneImporter/Common.h"
#include "VoxLib/Core/Format.h"
#include "VoxLib/Error/Error.h"
#include "VoxLib/IO/Resource.h"
#include "VoxScene/Scene.h"

// API namespace
namespace vox 
{
    
/** Defines export options for scene components */
enum ExportOpt
{
    ExportOpt_Begin,                       ///< Begin iterator for ExportOpt enumeration
    ExportOpt_Reference = ExportOpt_Begin, ///< Insert a reference (URI) to the original resource (even if modified)
    ExportOpt_CopyIfModified,              ///< Insert a copy only if the resource was modified (reference otherwise) 
    ExportOpt_Copy,                        ///< Generate a copy of the resource
    ExportOpt_Overwrite,                   ///< Overwrite existing content of resource if modified
    ExportOpt_None,                        ///< Ignore the resource (no reference/no copy)
    ExportOpt_End                          ///< End iterator for ExportOpt enumeration
};

/**
 * Vox scene file import / export module
 *
 * This module is compatible with the abstract scene import/export interface.
 */
class VSI_EXPORT VoxSceneFile : public SceneImporter, public SceneExporter
{
public:
    VoxSceneFile(std::shared_ptr<void> handle, bool isXml = true) : m_handle(handle), m_isXml(isXml) { }

	/** 
     * @brief Vox Scene File Exporter
     *
     * \b{Valid Options}
     *  - ExportVolume     : bool   [default=true]  | Specifies whether to export volume data
     *  - ExportCamera     : bool   [default=true]  | Specifies whether to export camera data
     *  - ExportTransfer   : bool   [default=true]  | Specifies whether to export transfer data
     *  - ExportLights     : bool   [default=true]  | Specifies whether to export lighting data
     *  - ExportFilm       : bool   [default=false] | Specifies whether to export the current film
     *  - ForceTransferMap : strnig [default=""]    | Forces export of the transfer as an image in the specified .ext
     *  - Compress         : string [default=""]    | Specifies compression types and order, supported
     *                                                compression modes include 'zlib', 'bzip2', and 'gzip'.
     *                                                Compression is applied from left to right treating
     *                                                the most common control characters as delimiters
     *
     * \b{Required Options}
     *  - None
     */
     void exporter(ResourceOStream & data, OptionSet const& options, Scene const& scene);

	/** 
     * @brief Vox Scene File Importer 
     *
     * \b{Valid Options}
     *  ImportVolume   : bool   [default=true]  | Specifies whether to import volume info
     *  ImportCamera   : bool   [default=true]  | Specifies whether to import camera info
     *  ImportTransfer : bool   [default=true]  | Specifies whether to import transfer info
     *  ImportLights   : bool   [default=true]  | Specifies whether to import lighting info
     *  ImportFilm     : bool   [default=false] | Specifies whether to import the current film
     *
     * \b{Required Options}
     *  - None
     */
	std::shared_ptr<Scene> importer(ResourceIStream & data, OptionSet const& options);

private:
    std::shared_ptr<void> m_handle; ///< Plugin handle to track this DLL's usage
    bool m_isXml;   ///< Determines whether to use XML or json encoding
};

}

// End definition
#endif // VSI_SCENE_FILE_H