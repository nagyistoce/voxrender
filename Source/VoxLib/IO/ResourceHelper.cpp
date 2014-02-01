    /* ===========================================================================

	Project: Uniform Resource IO 
    
	Description: Provides extended functionality relating to the Resource class

    Copyright (C) 2013 Lucas Sherman

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
#include "ResourceHelper.h"

// Include Dependencies
#include "VoxLib/Core/Logging.h"
#include "VoxLib/IO/Resource.h"
#include "VoxLib/Plugin/PluginManager.h"

// Boost XML Parser
#include <boost/property_tree/xml_parser.hpp>
#include <boost/filesystem.hpp>

// API Namespace
namespace vox 
{

namespace {
namespace filescope {
    
    // Operational importance rating
    enum Importance
    {
        Required,   ///< Throw an exception if not found
        Preferred,  ///< Issue a warning if not found
        Optional,   ///< Node is optional 
    };

    // Import module implementation
    class ConfigImporter
    {
    public:
        // --------------------------------------------------------------------
        //  Parse the resource data from an XML format into a property tree
        // --------------------------------------------------------------------
        ConfigImporter(String const& sourceFile) : m_node(&m_tree), m_identifier(sourceFile)
        {
            // Detect errors parsing the scene file's XML content
            try
            {
                std::ifstream source(sourceFile); // Access the file for reading

                if (!source) throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY,
                    format("Unable to open config file <%1%>", sourceFile), 
                    Error_NotFound);

                boost::property_tree::xml_parser::read_xml(source, m_tree);
            }
            catch(boost::property_tree::xml_parser::xml_parser_error & error)
            {
                throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY, 
                    format("%1% at line %2%", error.message(), error.line()), 
                    Error_Syntax);
            }
        }
            
        // --------------------------------------------------------------------
        //  Parse the property tree and composes the output scene object
        // --------------------------------------------------------------------
        void parse()
        {
            m_stack.reserve(6); // Reserve some space

            try
            {
                if (!push("Settings", Preferred)) return;
                    
                  loadLogSettings();
                  loadPluginSettings();

                pop();
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
        }

    private:
        typedef std::pair<String const, boost::property_tree::ptree*> Iterator; ///< Stack pointer

        boost::property_tree::ptree   m_tree;        ///< Scenefile tree
        boost::property_tree::ptree * m_node;        ///< Current node in tree (top of traversal stack)
        String const&                 m_identifier;  ///< Resource identifier for relative URLs
            
        std::vector<Iterator> m_stack; ///< Property tree traversal stack
        
        // --------------------------------------------------------------------
        //  Loads config information from the Logging element of the file
        // --------------------------------------------------------------------
        void loadLogSettings()
        {
            if (!push("Logging", Optional)) return;
                
                // Set the log filtering level
                auto filterStr = m_node->get<String>("Filter", "");
                boost::to_lower(filterStr);
                if (filterStr == "debug")           Logger::setFilteringLevel(Severity_Debug);
                else if (filterStr == "trace")      Logger::setFilteringLevel(Severity_Trace);
                else if (filterStr == "error")      Logger::setFilteringLevel(Severity_Error);
                else if (filterStr == "warning")    Logger::setFilteringLevel(Severity_Warning);
                else if (filterStr == "fatal")      Logger::setFilteringLevel(Severity_Fatal);
                else // :TODO: Attempt numerical cast to int, for user defined levels
                {
                    VOX_LOG_WARNING(Error_BadToken, VOX_LOG_CATEGORY, 
                        format("Unrecognized logging filter level in %1%, ignoring: Filter='%2%'", 
                            m_identifier, filterStr));
                }

                // Process and exclude module specifications
                if (push("ExcludeModules", Optional))
                {
                    BOOST_FOREACH (auto & child, *m_node)
                    {
                        if (child.first == "<xmlcomment>") continue; // Ignore comments

                        if (child.first == "Module")
                        {
                            auto category = child.second.get_value<String>();
                            Logger::addCategoryFilter(category);
                        }
                        else
                        {
                          VOX_LOG_WARNING(Error_BadToken, VOX_LOG_CATEGORY, 
                              format("Unrecognized child of Logger ExcludeModules in %1%, ignoring: name='%2%'", 
                                m_identifier, child.first));
                        }
                    }

                    pop();
                }

            pop();
        }
        
        // --------------------------------------------------------------------
        //  Loads config information from the Plugins element of the file
        // --------------------------------------------------------------------
        void loadPluginSettings()
        {
            if (!push("Plugins", Optional)) return;
                   
              // Load the specified plugin search directories
              if (push("SearchDirectories", Optional)) 
              {
                  auto rootPath = boost::filesystem::absolute(m_identifier, 
                      boost::filesystem::current_path()).remove_filename();

                  auto & pluginManager = PluginManager::instance();

                  BOOST_FOREACH (auto const& child, *m_node)
                  {
                      if (child.first == "Dir")
                      {
                          auto path = rootPath / child.second.get<String>("");

                          pluginManager.addPath(path.string());
                      }
                  }

                  pluginManager.search(false, true);

                  pop();
              }

              // Enable any of the authorized plugins
              if (push("Authorized", Optional))
              {
                  auto rootPath = boost::filesystem::absolute(m_identifier, 
                      boost::filesystem::current_path()).remove_filename();

                  auto & pluginManager = PluginManager::instance();

                  BOOST_FOREACH (auto const& child, *m_node)
                  {
                      if (child.first == "Plugin")
                      {
                          auto id = child.second.get<String>("");
                          std::vector<std::string> filters;
                          boost::algorithm::split(filters, id, 
                              boost::is_any_of("."), 
                              boost::algorithm::token_compress_on);
                          if (filters.size() != 2)
                              VOX_LOG_WARNING(Error_BadToken, VOX_LOG_CATEGORY,
                                format("Authorized plugin must be specified as Vendor.Name: %1%", id));
                          
                          auto plugin = pluginManager.findByNameVendor(filters[0], filters[1]);
                          if (plugin) pluginManager.load(plugin);
                      }
                  }

                  pop();
              }

            pop();
        }

        // --------------------------------------------------------------------
        //  Formats and throws a parse error with the specified error code
        // --------------------------------------------------------------------
        void parseError(ErrorCode code, std::string const& what)
        {
            throw Error(
                __FILE__, __LINE__, VOX_LOG_CATEGORY,
                format("%1% at \"%2%\" [%3%]", what, 
                        currentPath(), m_identifier),
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
                // Issue warning message about missing tag
                VOX_LOG_WARNING(Error_MissingData, VOX_LOG_CATEGORY, 
                    format("Node not found \"%1%\" [%2%]", name, m_identifier));
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
            
            if (m_stack.empty())
            {
                m_node = &m_tree;
            }
            else
            {
                m_node = m_stack.back().second; 
            }
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

} // namespace filescope
} // namespace anonymous

// ----------------------------------------------------------------------------
//  Utilizes the Resource library to copy URI request data into a buffer
// ----------------------------------------------------------------------------
std::string ResourceHelper::pull(ResourceId const& resource)
{
    ResourceIStream istr(resource);
    
    std::ostringstream outStr(std::ios::in|std::ios::binary);
    outStr << istr.rdbuf();

    return outStr.str();
}

// ----------------------------------------------------------------------------
//  Utilizes the Resource library to upload URI request data from a buffer
// ----------------------------------------------------------------------------
void ResourceHelper::push(ResourceId const& resource, std::string const& data)
{
    ResourceOStream ostr(resource);
    
    ostr << data;
}

// ----------------------------------------------------------------------------
//  Utilizes the Resource library to upload URI request data from a buffer
// ----------------------------------------------------------------------------
void ResourceHelper::move(ResourceId const& source, ResourceId const& destination)
{
    ResourceIStream istr(source);
    ResourceOStream ostr(destination);
    
    ostr << istr.rdbuf();
}

// ----------------------------------------------------------------------------
//  Utilizes the Resource library to upload URI request data from a buffer
// ----------------------------------------------------------------------------
void ResourceHelper::loadConfigFile(String const& configFile)
{
    auto importer = filescope::ConfigImporter(configFile);

    importer.parse();
}

} // namespace vox