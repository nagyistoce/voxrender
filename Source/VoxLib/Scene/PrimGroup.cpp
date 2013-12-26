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

namespace vox {
    
namespace {
namespace filescope {

    // Static member initialization
    Char const* typeId  = "PrimGroup";
    
    // ---------------------------------------------------------------------------- 
    //  Parses a primitive group from a property tree node
    // ---------------------------------------------------------------------------- 
    //void parseFunc(boost::property_tree::ptree const& data)
    //{
        //BOOST_FOREACH (auto const& node, data) Primitive::imprt(node);
    //}

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

// --------------------------------------------------------------------
//  Adds a new child node to this primitive group
// --------------------------------------------------------------------
void PrimGroup::add(std::shared_ptr<Primitive> child)
{
    m_children.push_front(child);

    child->setParent(shared_from_this());

    child->setDirty();

    setDirty();
}

// --------------------------------------------------------------------
//  Removes a child node from this primitive group
// --------------------------------------------------------------------
void PrimGroup::remove(std::shared_ptr<Primitive> child)
{
    m_children.remove(child);

    child->setParent(nullptr);

    setDirty();
}

// --------------------------------------------------------------------
//  Clears all elements from the primitive group
// --------------------------------------------------------------------
void PrimGroup::clear()
{ 
     m_children.clear(); 
     
     setDirty(); 
}

} // namespace vox