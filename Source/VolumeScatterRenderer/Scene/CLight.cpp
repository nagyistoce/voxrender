/* ===========================================================================

	Project: VoxRender - Device Side Camera

	Description: Defines a 3D Camera for use in rendering

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
#include "CCamera.h"

// Include Dependencies
#include "VoxLib/Scene/Light.h"

// API namespace
namespace vox
{

// --------------------------------------------------------------------
//  Constructs a light object for device side rendering 
// --------------------------------------------------------------------
CLight::CLight(Light const& light) :
    m_position(light.position()),
    m_color(light.color())
{
}

}