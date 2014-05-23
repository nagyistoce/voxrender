/* ===========================================================================

	Project: VoxLib

	Description: Data structure defining material properties of volume

    Copyright (C) 2012-2013 Lucas Sherman

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
#include "Material.h"

// Include Dependencies
#include "VoxScene/Transfer.h"

namespace vox {

std::shared_ptr<Material> Material::clone()
{
    auto result = create();
    
    *result = *this;

    return result;
}

std::shared_ptr<Material> Material::interp(std::shared_ptr<Material> k2, float f)
{
    auto result = create();
    
    result->opticalThickness = k2->opticalThickness * f + opticalThickness * (1.f - f);
    result->glossiness       = k2->glossiness * f       + glossiness * (1.f - f);
    result->emissiveStrength = k2->emissiveStrength * f + emissiveStrength * (1.f - f);
    result->emissive         = k2->emissive * f         + emissive * (1.f - f);
    result->diffuse          = k2->diffuse * f          + diffuse * (1.f - f);
    result->specular         =k2-> specular * f         + specular * (1.f - f);

    return result;
}

} // namespace vox