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
#include "VoxLib/Scene/PrimGroup.h"

namespace vox {

namespace {
namespace filescope {

    // Static member initialization
    Char const* planeTypeId  = "Plane";
    Char const* sphereTypeId = "Sphere";

} // namespace filescope 
} // namespace

// ----------------------------------------------------------------------------
//  Sets the parent group of a primitive 
// ----------------------------------------------------------------------------
void Primitive::setParent(std::shared_ptr<PrimGroup> parent) 
{ 
    m_parent = parent;
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

// ---------------------------------------------------------------------------- 
//  
// ---------------------------------------------------------------------------- 
void Plane::setNormal(Vector3f const& normal)
{
    m_normal = normal;

    setDirty();
}

// ---------------------------------------------------------------------------- 
//  
// ---------------------------------------------------------------------------- 
void Plane::setDistance(float distance)
{
    m_distance = distance;

    setDirty();
}

} // namespace vox