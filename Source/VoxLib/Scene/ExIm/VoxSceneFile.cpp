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
#include "VoxLib/Scene/RenderParams.h"
#include "VoxLib/Scene/Transfer.h"
#include "VoxLib/Scene/Volume.h"
#include "VoxLib/Scene/Material.h"

// Boost XML Parser
#include <boost/property_tree/xml_parser.hpp>

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
                        format("A read operation has silently failed [%1%]", m_displayName), 
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
                    scene.volume     = loadVolume();
                    scene.camera     = loadCamera();
                    scene.lightSet   = loadLights();
                    scene.transfer   = loadTransfer();
                    scene.parameters = loadParams();
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
            typedef std::pair<String const, boost::property_tree::ptree*> Iterator; ///< Stack pointer

            static int const versionMajor = 0; ///< Scene version major [potentially breaking]
            static int const versionMinor = 0; ///< Scene version minor [enforces non-breaking]

            boost::property_tree::ptree   m_tree;        ///< Scenefile tree
            boost::property_tree::ptree * m_node;        ///< Current node in tree (top of traversal stack)
            OptionSet const&              m_options;     ///< Import options
            String                        m_displayName; ///< Warning identifier for making log entries
            ResourceId const&             m_identifier;  ///< Resource identifier for relative URLs
            
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
                                                     versionMajor, versionMinor),
                                         __FILE__, __LINE__);
                    }
                }

                // Detect missing version info and issue warning
                catch(boost::property_tree::ptree_bad_path &)
                {
                    Logger::addEntry(vox::Severity_Warning, vox::Error_MissingData, VOX_LOG_CATEGORY,
                                     format("Scenefile is missing version info [%1%]", 
                                            m_displayName), 
                                     __FILE__, __LINE__);
                }
            }

            // --------------------------------------------------------------------
            //  Creates a camera object from the 'Camera' node of a scene file
            // --------------------------------------------------------------------
            std::shared_ptr<Camera> loadCamera()
            {
                if (!push("Camera", Preferred)) return nullptr;
                
                  // Instantiate default volume object
                  auto cameraPtr = executeImportDirectives().camera;
                  if (!cameraPtr) cameraPtr = std::make_shared<Camera>();
                  auto & camera = *cameraPtr;
                
                  // :TODO: This just overwrites imported parameters, should initialize to default
                  //        in Camera constructor and use import values as defaults for assignment

                  // Load direct camera projection parameters
                  camera.setApertureSize( m_node->get("ApertureSize", 0.0f) );
                  camera.setFieldOfView( m_node->get("FieldOfView", 60.0f) / 180.0f * (float)M_PI );
                  camera.setFocalDistance( m_node->get("FocalDistance", 0.0f) );

                  // Load camera orientation parameters
                  camera.setPosition( m_node->get("Position", Vector3f(0.0f, 0.0f, 0.0f)) );
                  camera.lookAt( m_node->get("Target", camera.position() + Vector3f(0.0f, 0.0f, 1.0f)) );

                  // Load camera film dimensions
                  camera.setFilmWidth( m_node->get("FilmWidth", 256) );
                  camera.setFilmHeight( m_node->get("FilmHeight", 256) );

                  // :TODO: Allow 2-3 specified control orients, compute resulting position
                  camera.setEye( m_node->get("Eye", Vector3f(0.0f, 0.0f, 1.0f)) );
                  camera.setRight( m_node->get("Right", Vector3f(1.0f, 0.0f, 0.0f)) );

                pop();

                return cameraPtr;
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

                  // Check for ambient level specification
                  lightSet.setAmbientLight( m_node->get<Vector3f>("Ambient", Vector3f(0.0f, 0.0f, 0.0f)) );

                  // Process standard light elements of the lighting subtree
                  auto bounds = m_node->equal_range("Light");
                  for (auto it = bounds.first; it != bounds.second; ++it)
                  {
                      auto & node = (*it).second;
                      
                      Vector3f color = node.get<Vector3f>("Color");

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
        
                  // Read inline volume parameter specifications
                  volume.setSpacing(m_node->get("Spacing", volume.spacing()));

                  // Do not allow any other parameter specifications here as they will 
                  // overwrite interdependent information (ie extent relates to data etc)
				  // and inline volume data specification is not supported

                pop();

                return volumePtr;
            }

            // --------------------------------------------------------------------
            //  Creates a parameters object from the 'Settings' node of a scene file
            // --------------------------------------------------------------------
            std::shared_ptr<RenderParams> loadParams()
            {
                if (!push("Settings", Preferred)) return nullptr;

                  // Instantiate default volume object
                  auto paramPtr = executeImportDirectives().parameters;
                  if (!paramPtr) paramPtr = std::make_shared<RenderParams>();
                  auto & parameters = *paramPtr;
        
                  // Read inline parameter specifications
                  parameters.setPrimaryStepSize(m_node->get("PrimaryStepSize", parameters.primaryStepSize()));
                  parameters.setShadowStepSize(m_node->get("ShadowStepSize", parameters.shadowStepSize()));
                  parameters.setOccludeStepSize(m_node->get("OccludeStepSize", parameters.occludeStepSize()));

                pop();

                return paramPtr;
            }

            // --------------------------------------------------------------------
            //  Creates a transfer object from the 'Transfer' node of a scene file
            // --------------------------------------------------------------------
            std::shared_ptr<Transfer> loadTransfer()
            {                
                if (!push("Transfer", Preferred)) return nullptr;

                  // Instantiate default transfer object
                  auto transferPtr = executeImportDirectives().transfer;
                  if (!transferPtr) transferPtr = std::make_shared<Transfer>();
                  auto & transfer = *transferPtr;
                  /*
                  // Transfer function resolution
                  String resolution = m_node->get("Resolution");
                  std::vector<String> dimensions;
                  boost::algorithm::split(
                    dimensions, 
                    resolution, 
                    boost::is_any_of(" ,\n\t\r"), 
                    boost::algorithm::token_compress_on
                    );
                  boost::lexical_cast<size_t>(dimensions[0]);
                  */

                  // Import any named materials 
                  auto materials = loadMaterials();

                  // Process transfer function nodes
                  if (push("Nodes", Preferred))
                  {
                      BOOST_FOREACH (auto & region, *m_node)
                      {
                          // Create a new node for insertion
                          auto node = std::make_shared<Node>();
                          transfer.addNode(node);

                          // Determine the node's material properties
                          auto materialOpt = region.second.get_optional<String>("Material");
                          if (materialOpt) // Check for name specification of material
                          {
                              auto matIter = materials.find(*materialOpt);
                              if (matIter != materials.end())
                              {
                                  node->setMaterial(matIter->second);
                              }
                              else parseError(Error_BadToken, format("Undefined material (%1%) used", *materialOpt));
                          }
                          else // load inline specification of material
                          {
                              auto material = std::make_shared<Material>();
                              material->setGlossiness( region.second.get("Glossiness", 0.0f) );
                              material->setOpticalThickness( region.second.get("Thickness", 0.0f) );
                              node->setMaterial(material);
                          }

                          // Determine the node's position
                          node->setPosition(0, region.second.get<float>("Density"));
                      }
                  }

                pop();

                return transferPtr;
            }

            // --------------------------------------------------------------------
            //  Processes the materials node of the transfer function
            // --------------------------------------------------------------------
            std::map<String, std::shared_ptr<Material>> loadMaterials()
            {
                std::map<String, std::shared_ptr<Material>> materials;

                if (push("Materials", Optional))
                {
                    BOOST_FOREACH(auto & materialNode, *m_node)
                    {
                        // Check for multiply defined material nodes
                        if (materials.find(materialNode.first) != materials.end())
                        {
                            parseError(Error_BadToken, format("Duplicate material defined (%1%)", materialNode.first));
                        }
                        
                        // Parse the material specification :TODO:
                        auto material = std::make_shared<Material>();
                        material->setGlossiness( m_node->get("Glossiness", 0.0f) );
                        material->setOpticalThickness( m_node->get("Thickness", 0.0f) );
                        materials[materialNode.first] = material;
                    }
           
                    pop();
                }

                return materials;
            }

            // --------------------------------------------------------------------
            //  Attempts to execute an import directive at the current node
            //  Component parameter specifies the desired component, allowing 
            //  only necessary scene components to be loaded using the generic
            //  exim 'Ignore' flag :TODO:
            // --------------------------------------------------------------------
            Scene executeImportDirectives(String const& component = "")
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
                  ResourceId resourceId = m_identifier + m_node->get<String>("Resource");
                  ResourceIStream resource(resourceId);

                  // Override mime-type with resource filename
                  // :TODO: Warn on mime-types/file extension mismatch
                  // :TODO: Allow override of extension in scenefile
                  // :TODO: Determine type from combo of file-ext and mime-type
                  String const extension = m_node->get(
                      "Type", resource.identifier().extractFileExtension()); 

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
                        format("Node not found \"%1%\" [%2%]", name, m_displayName), 
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
            //  Composes the current position string from the ptree stack
            // --------------------------------------------------------------------
            String currentPath()
            {
                String path;
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