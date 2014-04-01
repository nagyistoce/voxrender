/* ===========================================================================

	Project: VoxLib

	Description: Defines the basic primitive object element

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
#include "Primitive.h"

// Include Dependencies
#include "VoxLib/Core/Common.h"
#include "VoxScene/PrimGroup.h"

namespace vox {

namespace {
namespace filescope {

    // Static member initialization
    Char const* planeTypeId  = "Plane";
    Char const* sphereTypeId = "Sphere";

    static std::map<String, PrimImporter> importers; // Primitive parsers
    static boost::shared_mutex moduleMutex;          // Module access mutex for read-write locks

    // Importer registration
    class PrimitiveForceImport { public: PrimitiveForceImport() { 
        Primitive::registerImportModule(Plane::classTypeId(), Plane::imprt); } };
    static PrimitiveForceImport a;

} // namespace filescope 
} // namespace

// ----------------------------------------------------------------------------
//  Registers an import function for a given primitive type identifier
// ----------------------------------------------------------------------------
void Primitive::registerImportModule(String const& type, PrimImporter importer)
{
    // Acquire a read-lock on the modules for thread safety support
    boost::unique_lock<decltype(filescope::moduleMutex)> lock(filescope::moduleMutex);
    
    if (!importer) throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY, 
        "Attempted to register empty import module", Error_NotAllowed);

    filescope::importers[type] = importer; 
}

// --------------------------------------------------------------------
//  Removes a primitive import module
// --------------------------------------------------------------------
void Primitive::removeImportModule(String const& type)
{
    // Acquire a read-lock on the modules for thread safety support
    boost::unique_lock<decltype(filescope::moduleMutex)> lock(filescope::moduleMutex);
 
    auto iter = filescope::importers.find(type);
    if (iter != filescope::importers.end()) 
        filescope::importers.erase(iter);
}

// --------------------------------------------------------------------
//  Imports a primitive type registered in the importer map
// --------------------------------------------------------------------
std::shared_ptr<Primitive> Primitive::imprt(String const& type, boost::property_tree::ptree & node)
{
    // Acquire a read-lock on the modules for thread safety support
    boost::shared_lock<decltype(filescope::moduleMutex)> lock(filescope::moduleMutex);

	// Execute the register import module
    auto importer = filescope::importers.find(type);
    if (importer != filescope::importers.end())
    {
        return importer->second(node);
    }

    throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY, 
                "No import module found", Error_BadToken);
}

// ----------------------------------------------------------------------------
//  Converts the primitive into a text storage format
// ----------------------------------------------------------------------------
void Primitive::exprt(boost::property_tree::ptree & node)
{
}

// ----------------------------------------------------------------------------
//  Parses a primitive from a property tree structure
// ----------------------------------------------------------------------------
std::shared_ptr<Primitive> Plane::imprt(boost::property_tree::ptree & node)
{
    std::shared_ptr<Plane> plane = Plane::create();
    
    plane->m_distance = node.get("Distance", plane->m_distance);
    plane->m_normal   = node.get("Normal", plane->m_normal);

    return plane;
}

// ----------------------------------------------------------------------------
//  Converts the primitive into a text storage format
// ----------------------------------------------------------------------------
void Plane::exprt(boost::property_tree::ptree & node)
{
    Primitive::exprt(node);

    node.add("Distance", m_distance);
    node.add("Normal", m_normal);
}

// ----------------------------------------------------------------------------
//  Returns the UID for this primitive type
// ----------------------------------------------------------------------------
Char const* Plane::typeId() 
{ 
    return filescope::planeTypeId; 
}

// ----------------------------------------------------------------------------
//  Returns the UID for this primitive type
// ----------------------------------------------------------------------------          
Char const* Plane::classTypeId() 
{ 
    return filescope::planeTypeId; 
}  

// ----------------------------------------------------------------------------
//  Returns the UID for this primitive type
// ----------------------------------------------------------------------------   
Char const* Sphere::typeId()
{ 
    return filescope::sphereTypeId; 
}  
 
// ----------------------------------------------------------------------------
//  Returns the UID for this primitive type
// ----------------------------------------------------------------------------             
Char const* Sphere::classTypeId() 
{ 
    return filescope::sphereTypeId; 
}  

} // namespace vox