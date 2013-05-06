/* ===========================================================================

	Project: VoxRender - Scene

	Description: Defines the Scene class used by the Renderer

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
#ifndef VOX_SCENE_H
#define VOX_SCENE_H

// Include Dependencies
#include "VoxLib/Core/CudaCommon.h"
#include "VoxLib/Core/Types.h"
#include "VoxLib/IO/Resource.h"

// API namespace
namespace vox
{
    // Forward Decls
    class VOX_EXPORT Camera;
	class VOX_EXPORT Scene;
    class VOX_EXPORT Transfer;
    class VOX_EXPORT Volume;
    class VOX_EXPORT LightSet;

	/** 
	 * Scene File Importer
     *
     * This typedef specifies the format of a scene file importer. The VoxRender
     * library offers a built in system for controlling the import of scene information. 
     * Scene import/export modules are registered with the Scene class through a static
     * interface and associated with a regular expression. When an attempt is made to 
     * import a scene file through the Scene::import interface, the first SceneLoader whose
     * regular expression matches the parameter provided to load will be executed.
     *
     * This mechanism makes it easy to load scene files by overloading the load function
     * for different file types or transfer mechanisms. If a scene file is requested from 
     * a remote resource repository for instance, a specific importer can be setup to match 
     * the transfer protocol and perform the file transfer before executing a load function
     * again for the specific file format which was transferred.
     *
     * The default scene file loader for example is quite flexible, allowing certain scene 
     * elements within a scene file to be specified based on filename and type. The following 
     * [*.xml] format scene file for example would (when loaded with the default loader) load 
     * a volume data set from "MyVolume.raw" using the registered raw format volume importer. 
     * The options tags would also be passed to the registered ".raw" format importer. If the
     * filename does not contain a desirable extension, the <Force>ext</Force> tags can be
     * used to specify an alternative loader that should be used in place of the URL default.
     * If no such importer is registered, the [*.xml] importer would then throw an exception
     * to indicate that the file could not be loaded due to an unknown volume file type.
     *
     * @a "MySceneFile.xml":
     *
     * @code
     *
     * <Volume>
	 *     <Import>
     *         <Resource>Boxes.raw</Resource>
     *         <Force>raw</Force>
     *         <Options>
     *             <Endianess>little</Endianess>
     *             <BytesPerVoxel>1</BytesPerVoxel>
     *             <Size>[64 64 64 1]</Size>
     *             <Spacing>[1 1 1]</Spacing>
     *         </Options>
     *     </Import>
     * </Volume>
     *
     * @endcode
     *
     * There are no restriction on what a scene loader may throw, and any exceptions will be
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
     *  ::SceneExporter
	 */              
    typedef std::function<Scene(ResourceIStream & data, OptionSet const& options)> SceneImporter;

    /**
	 * Scene File Exporter
     *
     * This typedef specifies the format of a scene file exporter. The VoxRender
     * library offers a built in system for controlling the export of scene information. 
     * Scene import/export modules are registered with the Scene class through a static
     * interface and associated with a regular expression. When an attempt is made to 
     * export a scene file through the Scene::export interface, the first SceneExporter whose
     * extension matches the parameter provided to load will be executed.
     *
     * @sa
     *  ::SceneImporter
     */
    typedef std::function<void(ResourceOStream & data, OptionSet const& options, 
                               Scene const& scene)> SceneExporter;

	/** 
	 * @brief Scene Class
     *
     * This class enapsulates a set of elements required for rendering volume datasets.
     * There are also several static functions for managing the import and export of 
     * the scene data in an abstract manner.
	 */
	class VOX_EXPORT Scene
	{
	public:
		/**
		 * @brief Overload for imprt
         *
         * This function overloads the imprt function and automatically provides the 
         * identifier as the resource identifer and matchname and the matchname for 
         * selecting an Importer.
		 */
	    inline static Scene imprt(ResourceId const& identifier, OptionSet const& options = OptionSet())
        {
            return imprt(ResourceIStream(identifier), options);
        }

		/**
		 * @brief Imports a scene using the internal scene file loading mechanism.
		 * 
         * This function can be used as an abstract interface for importing scene
         * files. The data parameter is transparently passed to the selected 
         * importer. Any exceptions thrown by an importer will be propogated out.
         *
         * @sa 
         *  ::SceneExporter
         *
		 * @param data       [in] The data stream with the imported content
         * @param matchname  [in] The match string for the importer selection
         * @param options    [in] Optional string containing 'name: value' options 
         *
         * @returns The loaded scene object
         *
         * @throws
         *  ::Error No import module is defined which accepts the matchname
		 */
	    static Scene imprt(ResourceIStream & data, 
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
		 * @brief Exports a scene using the internal scene file export mechanism.
		 * 
         * This function can be used as an abstract interface for exporting scene
         * files. The data parameter is transparently passed to the selected 
         * exporter. Any exceptions thrown by an exporter will be propogated out.
         *
         * @sa 
         *  ::SceneExporter
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
		 * Registers a new scene importer with the specified regular expression object for 
         * matching. If an importer is already specified which has a conflicting regular 
         * expression matcher, the new importer will take precedence.
		 * 
		 * @param importer [in] The new scene importer to be registered
		 * @param matcher  [in] The regular expression for matching
		 */
        static void registerImportModule(String const& extension, SceneImporter importer);

		/**
		 * Registers a new scene exporter with the specified regular expression object for 
         * matching. If an exporter is already specified which has a conflicting regular 
         * expression matcher, the new exporter will take precedence.
		 * 
		 * @param loader  [in] The new scene exporter to be registered
		 * @param matcher [in] The regular expression for matching
		 */
        static void registerExportModule(String const& extension, SceneExporter exporter);

		/**
		 * Returns a reference to the list containing the active scene importers.
		 * 
		 * @return A const reference to the internal list containing scene importers.
		 */
		static std::map<String, SceneImporter> const& importers();

		/**
		 * Returns a reference to the list containing the active scene exporters.
		 * 
		 * @return A const reference to the internal list containing scene exporters.
		 */
		static std::map<String, SceneExporter> const& exporters();

        /** Releases handles to the scene's internal data components */
        void reset()
        {
            volume.reset(); 
            transfer.reset(); 
            lightSet.reset(); 
            camera.reset();
        }

        /** Logs warning for missing scene components */
        void issueWarningsForMissingHandles() const;

        std::shared_ptr<LightSet> lightSet; ///< Lighting Data
        std::shared_ptr<Transfer> transfer; ///< Transfer function
		std::shared_ptr<Volume>   volume;	///< Volume data
		std::shared_ptr<Camera>   camera;	///< Scene camera
	};
}

// End definition
#endif // VOX_SCENE_H