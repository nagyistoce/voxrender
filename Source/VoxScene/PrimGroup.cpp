/* ===========================================================================

	Project: VoxLib

	Description: Defines a collection of primitive elements

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
#include "PrimGroup.h"

#include "boost/property_tree/ptree.hpp"

namespace vox {
    
namespace {
namespace filescope {

    // Static member initialization
    Char const* typeId  = "PrimGroup";

    // Importer registration
    class PrimitiveForceImport { public: PrimitiveForceImport() { 
        Primitive::registerImportModule(PrimGroup::classTypeId(), PrimGroup::imprt); } };
    static PrimitiveForceImport a;

} // namespace filescope 
} // namespace

// ----------------------------------------------------------------------------
//  Returns the UID for this primitive type
// ----------------------------------------------------------------------------
Char const* PrimGroup::typeId() 
{ 
    return filescope::typeId; 
}

// ----------------------------------------------------------------------------
//  Returns the UID for this primitive type
// ----------------------------------------------------------------------------          
Char const* PrimGroup::classTypeId() 
{ 
    return filescope::typeId; 
}  

// ----------------------------------------------------------------------------
//  Parses a primitive from a property tree structure
// ----------------------------------------------------------------------------
std::shared_ptr<PrimGroup> PrimGroup::imprt(boost::property_tree::ptree & node)
{
    std::shared_ptr<PrimGroup> primGroup = PrimGroup::create();

    BOOST_FOREACH (auto & child, node)
    {
        primGroup->add(Primitive::imprt(child.first, child.second));
    }

    return primGroup;
}

// ----------------------------------------------------------------------------
//  Converts the primitive into a text storage format
// ----------------------------------------------------------------------------
void PrimGroup::exprt(boost::property_tree::ptree & node)
{
    BOOST_FOREACH (auto & object, m_children)
    {
        boost::property_tree::ptree cnode;
        object->exprt(cnode);
        node.add_child(object->typeId(), cnode);
    }
}

// --------------------------------------------------------------------
//  Returns a child primitive based on its id
// --------------------------------------------------------------------
std::shared_ptr<Primitive> PrimGroup::find(int id)
{
    BOOST_FOREACH (auto & prim, m_children)
        if (prim->id() == id) return prim;

    return nullptr;
}

// --------------------------------------------------------------------
//  Adds a new child node to this primitive group
// --------------------------------------------------------------------
void PrimGroup::add(std::shared_ptr<Primitive> child, bool suppress)
{
    m_children.push_front(child);

    child->setParent(shared_from_this());

    child->setDirty();

    setDirty();
    
    if (m_addCallback) m_addCallback(child, suppress);
}

// --------------------------------------------------------------------
//  Removes a child node from this primitive group
// --------------------------------------------------------------------
void PrimGroup::remove(std::shared_ptr<Primitive> child, bool suppress)
{
    auto entry = std::find(m_children.begin(), m_children.end(), child);
    if (entry == m_children.end()) return;

    m_children.erase(entry);

    child->setParent(nullptr);

    setDirty();
    
    if (m_removeCallback) m_removeCallback(child, suppress);
}

// --------------------------------------------------------------------
//  Clears all elements from the primitive group
// --------------------------------------------------------------------
void PrimGroup::clear()
{ 
     m_children.clear(); 
     
     setDirty(); 
}

// ----------------------------------------------------------------------------
//  Callback event modifier functions
// ----------------------------------------------------------------------------
void PrimGroup::onAdd(std::function<void(std::shared_ptr<Primitive>, bool)> callback)    
{ 
    m_addCallback = callback; 
}
void PrimGroup::onRemove(std::function<void(std::shared_ptr<Primitive>, bool)> callback) 
{ 
    m_removeCallback = callback; 
}
    
} // namespace vox