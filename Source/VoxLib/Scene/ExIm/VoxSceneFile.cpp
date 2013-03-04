/* ===========================================================================

	Project: VoxRender - Vox Scene File
    
	Description: Vox scene file import/export module

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

// Include Header
#include "VoxSceneFile.h"

// Include Dependencies
#include "VoxLib/Core/Debug.h"
#include "VoxLib/Core/Logging.h"
#include "VoxLib/Core/Types.h"
#include "VoxLib/Error/FileError.h"
#include "VoxLib/Scene/Camera.h"
#include "VoxLib/Scene/Film.h"
#include "VoxLib/Scene/Light.h"
#include "VoxLib/Scene/Transfer.h"
#include "VoxLib/Scene/Volume.h"

// Boost XML Parser
#include <boost/property_tree/xml_parser.hpp>

// Boost.Endian (conditional acceptance)
//#include <boost/endian/conversion.hpp>

// API namespace
namespace vox
{

// File scope namespace
namespace
{
    namespace filescope
    {
        // Operational importance rating
        enum Importance
        {
            Required,   ///< Throw an exception if not found
            Preferred,  ///< Issue a warning if not found
            Optional,   ///< Node is optional 
        };

        // Export module implementation
        class SceneExporter
        {
        public:
            // --------------------------------------------------------------------
            //  Parse the scene data into a boost::property_tree
            // --------------------------------------------------------------------
            SceneExporter(ResourceOStream & sink, OptionSet const& options, Scene const& scene)
            {
            }

            // --------------------------------------------------------------------
            //  Write the boost::property_tree as an XML file to the stream
            // --------------------------------------------------------------------
            void writeDataSceneFile()
            {
            }

        private:
            static int const versionMajor = 0; ///< Scene version major [potentially breaking]
            static int const versionMinor = 0; ///< Scene version minor [usually non-breaking]
        };

        // Import module implementation
        class SceneImporter
        {
        public:
            // --------------------------------------------------------------------
            //  Parse the resource data from an XML format into a property tree
            // --------------------------------------------------------------------
            SceneImporter(ResourceIStream & source, OptionSet const& options) : 
              m_options(options), m_node(&m_tree), m_identifier(source.identifier())
            {
                // Detect errors parsing the scene file's XML content
                try
                {
                    boost::property_tree::xml_parser::read_xml(source, m_tree);
                }
                catch(boost::property_tree::xml_parser::xml_parser_error & error)
                {
                    throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY, 
                        format("%1% at line %2%", error.message(), error.line()), 
                        Error_Syntax);
                }

                // Compose the resource identifier for log warning entries
                String filename = source.identifier().extractFileName();
                m_displayName = filename.empty() ? "UNKNOWN" : format("\"%1%\"", filename);

                // Detect IO stream errors
                if (!source)
                {
                    vox::Logger::addEntry(
                        vox::Severity_Warning, vox::Error_BadStream, VOX_LOG_CATEGORY, 
                        format("A read operation has silently failed [%1%]", m_displayName).c_str(), 
                        __FILE__, __LINE__);
                }
            }
            
            // --------------------------------------------------------------------
            //  Parse the property tree and composes the output scene object
            // --------------------------------------------------------------------
            Scene parseSceneFile()
            {
                m_stack.reserve(6); // Reserve some space

                Scene scene;

                try
                {
                    // Locate the root node
                    push("Scene", Required);

                    checkVersionInfo(); // Version check
                    
                    // Execute top level importer directives
                    scene = executeImportDirectives();

                    // Load scene components
                    scene.volume   = loadVolume();
                    scene.camera   = loadCamera();
                    scene.film     = loadFilm();
                    scene.lightSet = loadLights();
                }

                // Malformed data on a node read attempt
                catch(boost::property_tree::ptree_bad_data & error) 
                { 
                    parseError(Error_Syntax, format("%1% \"%2%\"", error.what(), 
                                                    error.data<char const*>())); 
                }

                // A required node was missing from the tree
                catch(boost::property_tree::ptree_bad_path & error) 
                { 
                    parseError(Error_MissingData, format("%1% \"%2%\"", error.what(), 
                                                         error.path<char const*>())); 
                }

                // An unknown parsing error has occurred while parsing the tree
                catch(boost::property_tree::ptree_error & error) 
                { 
                    parseError(Error_Unknown, error.what()); 
                }

                return scene;
            }

        private:
            typedef std::pair<std::string const, boost::property_tree::ptree*> Iterator; ///< Stack pointer

            static int const versionMajor = 0; ///< Scene version major [potentially breaking]
            static int const versionMinor = 0; ///< Scene version minor [usually non-breaking]

            boost::property_tree::ptree   m_tree;        ///< Scenefile tree
            boost::property_tree::ptree * m_node;        ///< Current node
            OptionSet const&              m_options;     ///< Import options
            String                        m_displayName; ///< Warning identifier
            ResourceId const&             m_identifier;  ///< Resource identifier
            
            std::vector<Iterator> m_stack; ///< Property tree traversal stack

            // --------------------------------------------------------------------
            //  Checks the scene file version info against the importer version
            // --------------------------------------------------------------------
            void checkVersionInfo()
            {
                try
                {
                    // Retrieve the version numbers from the property tree
                    int fileVersionMajor = m_node->get<int>("Version.Major");
                    int fileVersionMinor = m_node->get<int>("Version.Minor");

                    // Check for version mismatch and issue warning as necessary
                    if (versionMajor != fileVersionMajor || versionMinor != fileVersionMinor)
                    {
                        Logger::addEntry(vox::Severity_Warning, vox::Error_BadVersion, VOX_LOG_CATEGORY,
                                         vox::format("File version is \"%1%.%2%\" : expected \"%3%.%4%\"", 
                                                     fileVersionMajor, fileVersionMinor, 
                                                     versionMajor, versionMinor).c_str(),
                                         __FILE__, __LINE__);
                    }
                }

                // Detect missing version info and issue warning
                catch(boost::property_tree::ptree_bad_path &)
                {
                    Logger::addEntry(vox::Severity_Warning, vox::Error_MissingData, VOX_LOG_CATEGORY,
                                     format("Scenefile is missing version info [%1%]", 
                                            m_displayName).c_str(), 
                                     __FILE__, __LINE__);
                }
            }

            // --------------------------------------------------------------------
            //  Creates a camera object from the 'Camera' node of a scene file
            // --------------------------------------------------------------------
            std::shared_ptr<Camera> loadCamera()
            {
                if (!push("Camera", Preferred)) return nullptr;

                  // Instantiate default camera object
                  auto cameraPtr = std::make_shared<Camera>();
                  auto & camera = *cameraPtr;
                
                  // Load direct camera projection parameters
                  camera.setApertureSize( m_node->get("ApertureSize", 0.0f) );
                  camera.setFieldOfView( m_node->get("FieldOfView", 60.0f) / 180.0f * (float)M_PI );
                  camera.setFocalDistance( m_node->get("FocalDistance", 0.0f) );

                  // Load camera orientation parameters
                  camera.setPosition( m_node->get("Position", Vector3f(0.0f, 0.0f, 0.0f)) );
                  camera.lookAt( m_node->get("Target", camera.position() + Vector3f(0.0f, 0.0f, 1.0f)) );

                  // :TODO: Allow 2-3 specified orients, compute resulting position
                  camera.setEye( m_node->get("Eye", Vector3f(0.0f, 0.0f, 1.0f)) );
                  camera.setRight( m_node->get("Right", Vector3f(1.0f, 0.0f, 0.0f)) );

                pop();

                return cameraPtr;
            }

            // --------------------------------------------------------------------
            //  Creates a film object from the 'Film' node of a scene file
            // --------------------------------------------------------------------
            std::shared_ptr<Film> loadFilm()
            {
                if (!push("Film", Preferred)) return nullptr;
                
                  // Instantiate default film object
                  auto filmPtr = std::make_shared<Film>();
                  auto & film = *filmPtr;

                  film.setHeight( m_node->get<size_t>("Height", 512) );
                  film.setWidth( m_node->get<size_t>("Width", 512) );

                pop();

                return filmPtr;
            }
            
            // --------------------------------------------------------------------
            //  Creates a lights vector object from the 'Lights' node of a scene file
            // --------------------------------------------------------------------
            std::shared_ptr<LightSet> loadLights()
            {
                if (!push("Lights", Preferred)) return nullptr;

                  // Instantiate empty light object vector
                  auto lightSetPtr = std::make_shared<LightSet>();
                  auto & lightSet = *lightSetPtr;

                  // Process stanard light elements of the lighting subtree
                  auto bounds = m_node->equal_range("Light");
                  for (auto it = bounds.first; it != bounds.second; ++it)
                  {
                      auto & node = (*it).second;
                      
                      Vector3f color    = node.get<Vector3f>("Color");

                      auto light = lightSet.addLight();
                      light->setColor( ColorLabHdr(color[0], color[1], color[2]) );
                      light->setPosition( node.get<Vector3f>("Position") );
                  }

                pop();

                return lightSetPtr;
            }

            // --------------------------------------------------------------------
            //  Creates a volume object from the 'Volume' node of a scene file
            // --------------------------------------------------------------------
            std::shared_ptr<Volume> loadVolume()
            {
                if (!push("Volume", Preferred)) return nullptr;

                  // Instantiate default volume object
                  auto volumePtr = executeImportDirectives().volume;
                  if (!volumePtr) volumePtr = std::make_shared<Volume>();
                  auto & volume = *volumePtr;
        
                pop();

                return volumePtr;
            }

            // --------------------------------------------------------------------
            //  Creates a transfer object from the 'Transfer' node of a scene file
            // --------------------------------------------------------------------
            std::shared_ptr<Transfer> loadTransfer()
            {                
                if (!push("Transfer", Preferred)) return nullptr;

                  // Instantiate default transfer object
                  auto transferPtr = executeImportDirectives().transfer;

                pop();

                return transferPtr;
            }

            // --------------------------------------------------------------------
            //  Attempts to execute an import directive at the current node
            // --------------------------------------------------------------------
            Scene executeImportDirectives()
            {
                // Check for external importer specifications
                if (!push("Import")) return Scene();

                  // Build options set for the import operation
                  OptionSet optionSet;
                  if (push("Options")) 
                  {
                      BOOST_FOREACH(auto & option, *m_node)
                      {
                          optionSet.addOption(option.first, option.second.data());
                      }

                      pop(); 
                  }
               
                  // Attempt to import the imbeded resource
                  ResourceId resourceId = m_identifier + 
                      m_node->get<std::string>("Resource");
                  ResourceIStream resource(resourceId);

                  // Override mime-type with resource filename
                  // :TODO: Verify mime-types with filename extension
                  std::string const extension = m_node->get("Type", 
                      resource.identifier().extractFileExtension()); 

                pop();

                // Load the resource and execute the importer directive
                try
                {
                    return Scene::imprt(resource, optionSet, extension);
                }
                catch(Error & error)
                {
                    error.message = format("Failed to import \"%1%\" [%2%]", 
                                           resource.identifier().extractFileName(), 
                                           error.message);
                    throw;
                }
                catch(std::exception & error)
                {
                    throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY,
                                format("Failed to import \"%1%\" [%2%]", 
                                       resource.identifier().extractFileName(), 
                                       error.what()));
                }

                return Scene();
            }

            // --------------------------------------------------------------------
            //  Formats and throws a parse error with the specified error code
            // --------------------------------------------------------------------
            void parseError(ErrorCode code, std::string const& what)
            {
                throw Error(
                    __FILE__, __LINE__, VOX_LOG_CATEGORY,
                    format("%1% at \"%2%\" [%3%]", what, 
                           currentPath(), m_displayName),
                    code
                    );
            }
            
            // --------------------------------------------------------------------
            //  Steps into the specified child node of the property tree
            // --------------------------------------------------------------------
            bool push(char const* name, Importance importance = Optional)
            {
                // Push the child node onto the stack
                if (auto child = m_node->get_child_optional(name))
                {
                    m_node = &child.get();
                    m_stack.push_back( Iterator(name, m_node) );
                    return true;
                }

                // Issue warning / error message on failure
                if (importance == Preferred)
                {
                    // Issue warning message
                    vox::Logger::addEntry(
                        Severity_Warning, Error_MissingData, VOX_LOG_CATEGORY, 
                        format("Node not found \"%1%\" [%2%]", 
                               name, m_displayName).c_str(), 
                        __FILE__, __LINE__
                        );
                }
                else if (importance == Required)
                {
                    // Throw not found exception
                    parseError(Error_MissingData, format("Node not found \"%1%\"", name));
                }

                return false;
            }

            // --------------------------------------------------------------------
            //  Pops the current node from the traversal stack
            // --------------------------------------------------------------------
            void pop() 
            { 
                m_stack.pop_back(); 
                m_node = m_stack.back().second; 
            }

            // --------------------------------------------------------------------
            //  Returns the current position in the property tree traversal
            // --------------------------------------------------------------------
            std::string currentPath()
            {
                std::string path;
                BOOST_FOREACH(auto const& node, m_stack)
                {
                    path += node.first + '.';
                }
                path.pop_back();

                return path;
            }
        };
    }
}

// --------------------------------------------------------------------
//  Writes a vox scene file to the stream
// --------------------------------------------------------------------
void VoxSceneFile::exporter(ResourceOStream & sink, OptionSet const& options, Scene const& scene)
{
    // Parse scenefile object into boost::property_tree
    filescope::SceneExporter exportModule(sink, options, scene);

    // Write property tree to the stream
    exportModule.writeDataSceneFile();
}

// --------------------------------------------------------------------
//  Reads a vox scene file from the stream
// --------------------------------------------------------------------
Scene VoxSceneFile::importer(ResourceIStream & source, OptionSet const& options)
{
    // Parse XML format input file into boost::property_tree
    filescope::SceneImporter importModule(source, options);

    // Read property tree and load scene
    return importModule.parseSceneFile();
}

}