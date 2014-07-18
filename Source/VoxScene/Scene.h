/* ===========================================================================

	Project: VoxRender - Scene

	Description: Defines the Scene class used by the Renderer

    Copyright (C) 2012-2014 Lucas Sherman

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
#include "VoxScene/Common.h"
#include "VoxScene/Animator.h"

// Include Dependencies
#include "VoxLib/Core/CudaCommon.h"
#include "VoxLib/Core/Types.h"
#include "VoxLib/IO/Resource.h"

// API namespace
namespace vox
{
    // Forward Decls
    class VOXS_EXPORT Camera;
	class VOXS_EXPORT Scene;
    class VOXS_EXPORT Transfer;
    class VOXS_EXPORT Volume;
    class VOXS_EXPORT LightSet;
    class VOXS_EXPORT RenderParams;
    class VOXS_EXPORT PrimGroup;
    class VOXS_EXPORT TransferMap;
    class VOXS_EXPORT Animator;

	/** 
	 * Scene File Importer
     *
     * This is the interface class for a scene import module. The VoxRender
     * library offers a built in system for controlling the import of scene information. 
     * Scene import/export modules are registered with the Scene class through a static
     * interface and associated with a file type/extension. When an attempt is made to 
     * import a scene file through the Scene::import interface, the SceneLoader whose
     * type matches the parameter provided to imprt method will be executed.
     *
     * This mechanism makes it easy to load scene files by overloading the load function
     * for different file types or transfer mechanisms. In conjunction with the Resource
     * library, it is possible to develop flexible importers for various types in remote
     * locations with highly abstract code.
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
    class SceneImporter 
    { 
    public: 
        virtual std::shared_ptr<Scene> importer(ResourceIStream & data, OptionSet const& options) = 0;
                          
        virtual ~SceneImporter() { } 
    };

    /**
	 * Scene File Exporter
     *
     * This interface class for a scene export module. The VoxRender
     * library offers a built in system for controlling the export of scene information. 
     * Scene import/export modules are registered with the Scene class through a static
     * interface and associated with a file type/extension. When an attempt is made to 
     * export a scene file through the Scene::export interface, the SceneExporter whose
     * extension matches the extension provided will be executed.
     *
     * @sa
     *  ::SceneImporter
     */
    class SceneExporter 
    { 
    public: 
        virtual void exporter(
            ResourceOStream & data, OptionSet const& options, Scene const& scene) = 0; 
                          
        virtual ~SceneExporter() { } 
    };

	/** 
	 * @brief Scene Class
     *
     * This class enapsulates a set of elements required for rendering volume datasets.
     * There are also several static functions for managing the import and export of 
     * the scene data in an abstract manner.
	 */
	class VOXS_EXPORT Scene : public std::enable_shared_from_this<Scene>
    {
    public:
        /**  Instantiates a new scene object */
        static std::shared_ptr<Scene> create()
        {
            return std::make_shared<Scene>();
        }

		/**
		 * @brief Overload for imprt
         *
         * This function overloads the imprt function and automatically provides the 
         * identifier as the resource identifer and matchname and the matchname for 
         * selecting an Importer.
		 */
	    inline static std::shared_ptr<Scene> imprt(
            ResourceId const& identifier, OptionSet const& options = OptionSet())
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
		 * @param data      [in] The data stream with the imported content
         * @param options   [in] Additional options to be passed to the importer
         * @param extension [in] An override type/extension (otherwise it will be deduced
         *                       from the data parameter's URL file path component)
         *
         * @returns The loaded scene object
         *
         * @throws
         *  ::Error No import module is defined which accepts the matchname
		 */
	    static std::shared_ptr<Scene> imprt(
            ResourceIStream & data, 
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
		 * @param data      [in] The data stream for the exported content
         * @param options   [in] Additional options to be passed to the exporter
         * @param extension [in] An override type/extension (otherwise it will be deduced
         *                       from the data parameter's URL file path component)
         *
         * @throws
         *  ::Error No export module is defined which accepts the matchname
         *  ::Error The export module returned an error
		 */
		void exprt(ResourceOStream & data, 
                   OptionSet const&  options   = OptionSet(), 
                   String const&     extension = String()) const; 

		/**`
		 * Registers a new scene importer with the specified extension. If an importer is already 
         * specified which has a conflicting extension, it will be overridden.
		 * 
		 * @param importer [in] The new scene importer to be registered
		 * @param matcher  [in] The file type/extension for matching
		 */
        static void registerImportModule(String const& extension, std::shared_ptr<SceneImporter> importer);

		/**
		 * Registers a new scene exporter with the specified regular expression object for 
         * matching. If an exporter is already specified which has a conflicting regular 
         * expression matcher, the new exporter will take precedence.
		 * 
		 * @param loader  [in] The new scene exporter to be registered
		 * @param matcher [in] The file type/extension for matching
		 */
        static void registerExportModule(String const& extension, std::shared_ptr<SceneExporter> exporter);

        /** Removes a scene import module for a specified extension, or all by default */
        static void removeImportModule(std::shared_ptr<SceneImporter> importer, String const& extension = "");

        /** Removes a scene export module for a specified extension, or all by default */
        static void removeExportModule(std::shared_ptr<SceneExporter> exporter, String const& extension = "");
        
        /** Removes a scene import module */
        static void removeImportModule(String const& extension);

        /** Removes a scene export module */
        static void removeExportModule(String const& extension);

        /** Constructs a keyframe for the current state of a scene */
        std::shared_ptr<KeyFrame> generateKeyFrame();

        /** Clones the scene (optionally into an preallocated scene) */
        std::shared_ptr<Scene> clone(std::shared_ptr<Scene> inPlace = nullptr);

        /** Returns true if any of the scene components are dirty */
        bool isDirty() const;

        /** 
         * Pads the scene to ensure that it is viable for rendering 
         *
         * Padding involves allocating default scene components for any null members.
         */
        void pad();

        /** Sets the scene change event callback */
        void onSceneChanged(std::function<void(Scene&,void*)> callback);

        /** Locks the scene for editing */
        std::shared_ptr<void> lock(void * userInfo = nullptr);

        std::shared_ptr<RenderParams> parameters;   ///< Rendering parameters
        std::shared_ptr<LightSet>     lightSet;     ///< Lighting Data
        std::shared_ptr<PrimGroup>    clipGeometry; ///< Clipping objects
        std::shared_ptr<Transfer>     transfer;     ///< Transfer function
        std::shared_ptr<TransferMap>  transferMap;  ///< Transfer function map
		std::shared_ptr<Volume>       volume;	    ///< Volume data
		std::shared_ptr<Camera>       camera;	    ///< Scene camera
        std::shared_ptr<Animator>     animator;     ///< Scene animation data
        
    public:
        Scene();
        ~Scene();

    private:
        class Impl;
        Impl * m_pImpl;
    };
}

// End definition
#endif // VOX_SCENE_H