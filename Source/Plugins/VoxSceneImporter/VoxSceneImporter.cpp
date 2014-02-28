/* ===========================================================================

    Project: Vox Scene Importer - Module definition for scene importer

    Description: A vox scene file importer module

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

// Include Header
#include "VoxSceneImporter.h"

// Include Dependencies
#include "Strings.h"
#include "VoxLib/Bitmap/Bitmap.h"
#include "VoxLib/Core/Debug.h"
#include "VoxLib/Core/Logging.h"
#include "VoxLib/Core/Types.h"
#include "VoxLib/Error/FileError.h"
#include "VoxScene/Camera.h"
#include "VoxScene/Film.h"
#include "VoxScene/Light.h"
#include "VoxScene/RenderParams.h"
#include "VoxScene/Transfer.h"
#include "VoxScene/Volume.h"
#include "VoxScene/Material.h"
#include "VoxScene/PrimGroup.h"

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

        // Converts a material to a property tree representation
        boost::property_tree::ptree toPtree(std::shared_ptr<Material> material)
        {
            boost::property_tree::ptree node;

            node.add(M_GLOSSINESS, material->glossiness);
            node.add(M_THICKNESS, material->opticalThickness);
            node.add(M_DIFFUSE, Vector3u(material->diffuse));
            node.add(M_SPECULAR, Vector3u(material->specular));
            node.add(M_EMISSIVE, Vector3u(material->emissive));

            return node;
        }

        // Constructs a material from a property tree representation
        std::shared_ptr<Material> toMaterial(boost::property_tree::ptree & node)
        {
            auto material = Material::create();

            material->glossiness       = node.get(M_GLOSSINESS, material->glossiness);
            material->opticalThickness = node.get(M_THICKNESS, material->opticalThickness);
            material->diffuse          = node.get(M_DIFFUSE, Vector3u(material->diffuse));
            material->specular         = node.get(M_SPECULAR, Vector3u(material->specular));
            material->emissive         = node.get(M_EMISSIVE, Vector3u(material->emissive));
            material->emissiveStrength = node.get("EmissiveStrength", material->emissiveStrength);

            return material;
        }

        // Export module implementation
        class SceneExporter
        {
        public:
            // --------------------------------------------------------------------
            //  Parse the scene data into a boost::property_tree
            // --------------------------------------------------------------------
            SceneExporter(ResourceOStream & sink, OptionSet const& options, Scene const& scene) :
                m_scene(scene), m_options(options), m_sink(sink)
            {
            }

            // --------------------------------------------------------------------
            //  Write the boost::property_tree as an XML file to the stream
            // --------------------------------------------------------------------
            void writeSceneFile()
            {
                // Write the version info to the tree
                m_tree.add("Scene.Version.Major", versionMajor);
                m_tree.add("Scene.Version.Minor", versionMinor);

                // Write the scene information
                if (m_options.lookup("ExportCamera", true))   writeCamera();
                if (m_options.lookup("ExportVolume", true))   writeVolume();
                if (m_options.lookup("ExportLights", true))   writeLighting();
                if (m_options.lookup("ExportTransfer", true)) writeTransfer();
                if (m_options.lookup("ExportClipGeo", true))  writeClipGeometry();
                if (m_options.lookup("ExportParams", true))   writeParams();

                // Write the compiled XML data to the output stream
                boost::property_tree::xml_writer_settings<char> settings('\t', 1);
                boost::property_tree::xml_parser::write_xml(m_sink, m_tree, settings);
            }

        private:
            // --------------------------------------------------------------------
            //  Write the camera settings to the property tree
            // --------------------------------------------------------------------
            void writeCamera()
            {
                if (!m_scene.camera) return;

                boost::property_tree::ptree node;
                auto camera = m_scene.camera;
                
                node.add(C_APERTURE, camera->apertureSize());
                node.add(C_FOV, camera->fieldOfView() * 180.0f / (float)M_PI);
                node.add(C_FOCAL_DIST, camera->focalDistance());
                node.add(C_FWIDTH, camera->filmWidth());
                node.add(C_FHEIGHT, camera->filmHeight());
                node.add(C_POSITION, camera->position());
                node.add(C_UP, camera->up());
                node.add(C_RIGHT, camera->right());

                m_tree.add_child("Scene.Camera", node);
            }
            
            // --------------------------------------------------------------------
            //  Write the volume settings to the property tree (uses .raw format)
            // --------------------------------------------------------------------
            void writeVolume() 
            {
                if (!m_scene.volume) return;

                boost::property_tree::ptree node;
                auto const COMP_FORMAT = "gzip";
                auto volume = m_scene.volume;
                
                // Compose the volume format options
                OptionSet options;
                options.addOption("Compression", COMP_FORMAT);

                // Write the raw format volume to disk
                auto baseUrl  = m_sink.identifier();
                auto filename = baseUrl.extractFileName();
                filename      = filename.substr(0, filename.find_last_of('.')) + ".raw"; 
                auto volUrl   = baseUrl.applyRelativeReference(filename);
                m_scene.exprt(volUrl, options);

                // Imbed the import directives
                node.add("Import.Options.Size", volume->extent());
                node.add("Import.Options.Spacing", volume->spacing());
                node.add("Import.Options.Type", Volume::typeToString(volume->type()));
                node.add("Import.Options.Offset", volume->offset());
                node.add("Import.Options.Compression", COMP_FORMAT);
                node.add("Import.Endianess", "little");
                node.add("Import.Resource", filename);

                m_tree.add_child("Scene.Volume", node);
            }

            // --------------------------------------------------------------------
            //  Write the lighting settings to the property tree
            // --------------------------------------------------------------------
            void writeLighting()
            {
                if (!m_scene.lightSet) return;

                boost::property_tree::ptree node;
                auto lights = m_scene.lightSet;
                
                // Check for ambient level specification
                node.add("Ambient", lights->ambientLight());
                BOOST_FOREACH (auto & light, lights->lights())
                {
                    boost::property_tree::ptree cNode;

                    cNode.add("Color", light->color());
                    cNode.add("Position", light->position());

                    node.add_child("Light", cNode);
                }

                m_tree.add_child("Scene.Lights", node);
            }

            // --------------------------------------------------------------------
            //  Write the transfer settings to the property tree
            // --------------------------------------------------------------------
            void writeTransfer()
            {
                if (!m_scene.transfer && !m_scene.transferMap) return;

                boost::property_tree::ptree node;
                auto transfer = m_scene.transfer;
                
                if (!transfer) // Transfer map only (or no transfer)
                {
                    // :TODO: Export raw image data
                }
                else if (transfer->type() == Transfer1D::typeID()) writeTransfer1D(node);
                else if (transfer->type() == Transfer2D::typeID()) writeTransfer2D(node);
                else
                {
                    throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY, "Unrecognized transfer function type", Error_BadFormat);
                }

                m_tree.add_child("Scene.Transfer", node);
            }
            
            // --------------------------------------------------------------------
            //  Write the transfer settings to the property tree (1D)
            // --------------------------------------------------------------------
            void writeTransfer1D(boost::property_tree::ptree & node)
            {
                auto transfer1D = dynamic_cast<Transfer1D*>(m_scene.transfer.get());

                node.add(T_TYPE, 1);
                node.add(T_RESOLUTION, transfer1D->resolution()[0]);

                BOOST_FOREACH (auto & point, transfer1D->nodes())
                {
                    boost::property_tree::ptree cNode = toPtree(point->material);
                    cNode.add("Density", point->density);
                    node.add_child("Nodes.Node", cNode);
                }
            }

            // --------------------------------------------------------------------
            //  Write the transfer settings to the property tree (2D)
            // --------------------------------------------------------------------
            void writeTransfer2D(boost::property_tree::ptree & node)
            {
                auto transfer2D = dynamic_cast<Transfer2D*>(m_scene.transfer.get());

                node.add(T_TYPE, 2);
                auto res = transfer2D->resolution();
                node.add(T_RESOLUTION, Vector2u(res[0], res[1]));

                BOOST_FOREACH (auto & quad, transfer2D->quads())
                {
                    boost::property_tree::ptree cNode;

                    cNode.add(Q_POSITION, quad->position);
                    cNode.add(Q_HEIGHTS, quad->heights);
                    cNode.add(Q_WIDTHS, quad->widths);
                    
                    cNode.add_child("UL", toPtree(quad->materials[Quad::Node_UL]));
                    cNode.add_child("LL", toPtree(quad->materials[Quad::Node_LL]));
                    cNode.add_child("UR", toPtree(quad->materials[Quad::Node_UR]));
                    cNode.add_child("LR", toPtree(quad->materials[Quad::Node_LR]));

                    node.add_child("Quads.Quad", cNode);
                }
            }

            // --------------------------------------------------------------------
            //  Write the transfer settings to the property tree (3D)
            // --------------------------------------------------------------------
            void writeTransfer3D(boost::property_tree::ptree & node)
            {
            }

            // --------------------------------------------------------------------
            //  Write the render parameter settings to the property tree
            // --------------------------------------------------------------------
            void writeParams()
            {
                if (!m_scene.parameters) return;

                boost::property_tree::ptree node;
                auto settings = m_scene.parameters;
                
                node.add(P_STEP_PRIMARY, settings->primaryStepSize());
                node.add(P_STEP_SHADOW,  settings->shadowStepSize());

                m_tree.add_child("Scene.Settings", node);
            }

            // --------------------------------------------------------------------
            //  
            // --------------------------------------------------------------------
            void writeAnimator()
            {
                if (!m_scene.animator) return;

                boost::property_tree::ptree node;

                node.add(P_ANI_FRAME, m_scene.animator->framerate());

                m_tree.add_child("Scene.Animator", node);
            }

            // --------------------------------------------------------------------
            //  Write the clipping geometry settings to the property tree
            // --------------------------------------------------------------------
            void writeClipGeometry()
            {
                if (!m_scene.clipGeometry) return;

                boost::property_tree::ptree node;
                auto graph = m_scene.clipGeometry;
                graph->exprt(node);

                m_tree.add_child("Scene.ClipGeometry", node);
            }

        private:
            static int const versionMajor = 0; ///< Scene version major [potentially breaking]
            static int const versionMinor = 0; ///< Scene version minor [usually non-breaking]
            
            boost::property_tree::ptree m_tree;     ///< Scenefile tree
            OptionSet const&            m_options;  ///< Export options

            Scene const& m_scene; // Scene
            ResourceOStream & m_sink;
        };

        // Import module implementation
        class SceneImporter
        {
        public:
            // --------------------------------------------------------------------
            //  Parse the resource data from an XML format into a property tree
            // --------------------------------------------------------------------
            SceneImporter(ResourceIStream & source, OptionSet const& options, std::shared_ptr<void> handle) : 
              m_options(options), m_node(&m_tree), m_identifier(source.identifier()), m_handle(handle)
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
                    if (m_options.lookup("ImportVolume", true)) scene.volume       = loadVolume();
                    if (m_options.lookup("ImportCamera", true)) scene.camera       = loadCamera();
                    if (m_options.lookup("ImportLights", true)) scene.lightSet     = loadLights();
                    if (m_options.lookup("ImportParams", true)) scene.parameters   = loadParams();
                    if (m_options.lookup("ImportTransfer", true)) loadTransfer(scene);
                    if (m_options.lookup("ImportAnimator", true)) loadAnimator();
                    scene.clipGeometry = loadClipGeometry();
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

            std::shared_ptr<void> & m_handle;

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
                  if (!cameraPtr) cameraPtr = Camera::create();
                  auto & camera = *cameraPtr;
                
                  // Load direct camera projection parameters
                  camera.setApertureSize(m_node->get(C_APERTURE, camera.apertureSize()));
                  camera.setFieldOfView(m_node->get(C_FOV, 60.0f) / 180.0f * (float)M_PI);
                  camera.setFocalDistance(m_node->get(C_FOCAL_DIST, camera.focalDistance()));

                  // Load camera film dimensions
                  camera.setFilmWidth( m_node->get(C_FWIDTH,  camera.filmWidth()) );
                  camera.setFilmHeight(m_node->get(C_FHEIGHT, camera.filmHeight()));

                  // Load camera position
                  camera.setPosition(m_node->get(C_POSITION, camera.position()));

                  // Load camera orientation
                  auto target = m_node->get_optional<Vector3f>(C_TARGET);
                  auto right  = m_node->get_optional<Vector3f>(C_RIGHT);
                  auto up     = m_node->get_optional<Vector3f>(C_UP);
                  auto eye    = m_node->get_optional<Vector3f>(C_EYE);
                  Vector3f fEye;
                  Vector3f fUp;
                  if (target)           fEye = target.get() - camera.position();
                  else if (up && right) fEye = Vector3f::cross(up.get(), right.get());
                  else if (eye)         fEye = eye.get();
                  else                  fEye = Vector3f(0.0f, 0.0f, 1.0f);
                  if (right)            fUp = Vector3f::cross(right.get(), fEye);
                  else if (up)          fUp = up.get();
                  else                  fUp = Vector3f(0.0f, 1.0f);
                  auto fRight = Vector3f::cross(fEye, fUp);
                  camera.lookAt(camera.position() + fEye, Vector3f::cross(fRight, fEye));

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
                  auto lightSetPtr = LightSet::create();
                  auto & lightSet = *lightSetPtr;

                  // Check for ambient level specification
                  lightSet.setAmbientLight( m_node->get<Vector3f>("Ambient", Vector3f(0.0f, 0.0f, 0.0f)) );

                  // Process standard light elements of the lighting subtree
                  auto bounds = m_node->equal_range("Light");
                  for (auto it = bounds.first; it != bounds.second; ++it)
                  {
                      auto & node = (*it).second;
                      
                      auto light = Light::create();
                      light->setColor(node.get<Vector3f>("Color"));
                      light->setPosition(node.get<Vector3f>("Position"));
                      lightSet.add(light);
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
                  if (!volumePtr) volumePtr = Volume::create();
                  auto & volume = *volumePtr;
        
                  // Read inline volume parameter specifications
                  volume.setSpacing(m_node->get(V_SPACING, volume.spacing()));
                  volume.setOffset(m_node->get(V_OFFSET, volume.offset()));

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
                  if (!paramPtr) paramPtr = RenderParams::create();
                  auto & parameters = *paramPtr;
        
                  // Read inline parameter specifications
                  parameters.setPrimaryStepSize(m_node->get(P_STEP_PRIMARY, parameters.primaryStepSize()));
                  parameters.setShadowStepSize(m_node->get(P_STEP_SHADOW, parameters.shadowStepSize()));

                pop();

                return paramPtr;
            }

            // --------------------------------------------------------------------
            //  Creates a transfer object from the 'Transfer' node of a scene file
            // --------------------------------------------------------------------
            void loadTransfer(Scene & scene)
            {                
                if (!push("Transfer", Optional)) { loadTransferMap(scene); return; }

                  // Execute a transfer function import directive if specified
                  auto transferImprt = executeImportDirectives();
                  if (transferImprt.transfer || transferImprt.transferMap) 
                  { 
                      scene.transferMap = transferImprt.transferMap;
                      scene.transfer = transferImprt.transfer;
                      return;
                  }
                  // End

                  // Load the transfer function specification, only supported
                  // xml types are the built in TransferXX classes and raw images
                  size_t type = m_node->get<size_t>("Type", 1);
                  switch (type)
                  {
                  case 1: scene.transfer = loadTransfer1D(); break;
                  case 2: scene.transfer = loadTransfer2D(); break;
                  case 3: scene.transfer = loadTransfer3D(); break;
                  default:
                      parseError(Error_BadToken, format("Unrecognized transfer function type: %1%", type));
                      break;
                  }

                pop();
            }
            
            // --------------------------------------------------------------------
            // --------------------------------------------------------------------
            void loadTransferMap(Scene & scene)
            {
                if (!push("TransferMap", Optional)) return;

                  // Execute a transfer function import directive if specified
                  auto transferImprt = executeImportDirectives();
                  if (transferImprt.transfer || transferImprt.transferMap) 
                  { 
                      scene.transferMap = transferImprt.transferMap;
                      return;
                  }

                  scene.transferMap = TransferMap::create();
                  auto & tmap = *scene.transferMap.get();

                  // Load the raw image files for the transfer map
                  auto opacity  = Bitmap::imprt(m_node->get<String>("Opacity"));
                  auto diffuse  = Bitmap::imprt(m_node->get<String>("Diffuse"));
                  auto specular = Bitmap::imprt(m_node->get<String>("Specular"));
                  auto emissive = Bitmap::imprt(m_node->get<String>("Emissive"));

                  // Generate the transfer function map 
                  // :TODO: Shouldn't be necessary, bug in higher level user of imprt
                  scene.transferMap = TransferMap::create();
                  scene.transfer->generateMap(scene.transferMap);

                pop();
            }

            // --------------------------------------------------------------------
            //  Loads a 1 dimensional transfer function specification
            // --------------------------------------------------------------------
            std::shared_ptr<Transfer> loadTransfer1D()
            {
                auto transfer = Transfer1D::create();

                transfer->setResolution(m_node->get<size_t>(T_RESOLUTION, 256)); // Resolution
                auto materials = loadMaterials(); // Import any referenced materials 

                // Process transfer function nodes
                if (push("Nodes", Preferred))
                {
                    BOOST_FOREACH (auto & region, *m_node)
                    {
                        // Create a new node for insertion
                        auto node = Node::create(
                            region.second.get<float>("Density"));
                        transfer->addNode(node);

                        // Determine the node's material properties
                        auto materialOpt = region.second.get_optional<String>("Material");
                        if (materialOpt) // Check for name specification of material
                        {
                            auto matIter = materials.find(*materialOpt);
                            if (matIter != materials.end())
                            {
                                node->material = matIter->second;
                            }
                            else parseError(Error_BadToken, format("Undefined material (%1%) used", *materialOpt));
                        }
                        else node->material = toMaterial(region.second);
                    }

                    pop();
                }

                return transfer;
            }
            
            // --------------------------------------------------------------------
            //  Loads a 2 dimensional transfer function specification
            // --------------------------------------------------------------------
            std::shared_ptr<Transfer> loadTransfer2D()
            {
                auto transfer = Transfer2D::create();

                transfer->setResolution(m_node->get(T_RESOLUTION, Vector2u(256, 128))); // Resolution
                auto materials = loadMaterials(); // Import any referenced materials 

                // Process transfer function nodes
                if (push("Quads", Preferred))
                {
                    BOOST_FOREACH (auto & region, *m_node)
                    {
                        auto quad = Quad::create();
                        quad->position = region.second.get(Q_POSITION, quad->position);
                        quad->heights  = region.second.get(Q_HEIGHTS, quad->heights);
                        quad->widths   = region.second.get(Q_WIDTHS, quad->widths);
                        if (auto node = region.second.get_child_optional("UL")) quad->materials[Quad::Node_UL] = toMaterial(*node);
                        if (auto node = region.second.get_child_optional("UR")) quad->materials[Quad::Node_UR] = toMaterial(*node);
                        if (auto node = region.second.get_child_optional("LL")) quad->materials[Quad::Node_LL] = toMaterial(*node);
                        if (auto node = region.second.get_child_optional("LR")) quad->materials[Quad::Node_LR] = toMaterial(*node);
                        transfer->addQuad(quad);
                    }

                    pop();
                }

                return transfer;
            }
                        
            // --------------------------------------------------------------------
            //  Loads a 3 dimensional transfer function specification
            // --------------------------------------------------------------------
            std::shared_ptr<Transfer> loadTransfer3D()
            {
                /*auto transfer = Transfer3D::create();

                transfer->setResolution(m_node->get(T_RESOLUTION, Vector3u(128, 64, 8))); // Resolution
                auto materials = loadMaterials(); // Import any referenced materials 

                return transfer;*/

                return nullptr;
            }

            // --------------------------------------------------------------------
            //  Creates a GeometrySet for clipping from the transfer node of a scene file
            // --------------------------------------------------------------------
            std::shared_ptr<PrimGroup> loadClipGeometry()
            {
                if (!push("ClipGeometry", Preferred)) return nullptr;

                    // Instantiate default volume object
                    auto geoPtr = executeImportDirectives().clipGeometry;
                    if (!geoPtr) geoPtr = PrimGroup::create();
                    auto & geometrySet = *geoPtr;

                    // Parse inline geometry specifications
                    geoPtr = PrimGroup::imprt(*m_node);

                pop();

                return geoPtr;
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
                        
                        materials[materialNode.first] = toMaterial(materialNode.second);
                    }
           
                    pop();
                }

                return materials;
            }
            
            // --------------------------------------------------------------------
            //  Loads animation and keyframe information
            // --------------------------------------------------------------------
            std::shared_ptr<Animator> loadAnimator()
            {
                if (!push("Animator", Preferred)) return nullptr;

                    // Instantiate default volume object
                    auto animatorPtr = executeImportDirectives().animator;
                    if (!animatorPtr) animatorPtr = Animator::create();
                    auto & animator = *animatorPtr;

                    // Load the framerate specification
                    auto framerate = m_node->get(P_ANI_FRAME, animator.framerate());
                    if (framerate > 120) 
                    {
                        framerate = 120;
                        VOX_LOG_WARNING(Error_Range, VSI_LOG_CATEGORY, 
                            "Framerate clipped to maximum of 120Hz");
                    }
                    animator.setFramerate(framerate);

                    return animatorPtr;

                pop();
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
    exportModule.writeSceneFile();
}

// --------------------------------------------------------------------
//  Reads a vox scene file from the stream
// --------------------------------------------------------------------
Scene VoxSceneFile::importer(ResourceIStream & source, OptionSet const& options)
{
    // Parse XML format input file into boost::property_tree
    filescope::SceneImporter importModule(source, options, m_handle);

    // Read property tree and load scene
    return importModule.parseSceneFile();
}

}