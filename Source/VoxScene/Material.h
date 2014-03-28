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

// Begin definition
#ifndef VOX_MATERIAL_H
#define VOX_MATERIAL_H

// Include Dependencies
#include "VoxScene/Common.h"
#include "VoxLib/Core/CudaCommon.h"
#include "VoxLib/Core/Geometry/Color.h"

// API Namespace
namespace vox
{
    /** Defines the material properties of a volume. */
    class VOXS_EXPORT Material
    {
    public:
        static std::shared_ptr<Material> create() 
        { 
            return std::shared_ptr<Material>(new Material()); 
        }

        bool operator==(Material const& rhs)
        {
            return opticalThickness == rhs.opticalThickness &&
                   glossiness       == rhs.glossiness &&
                   emissiveStrength == rhs.emissiveStrength &&
                   specular         == rhs.specular &&
                   diffuse          == rhs.diffuse &&
                   emissive         == rhs.emissive;
        }

        bool operator!=(Material const& rhs) { return !((*this)==(rhs)); }

        float opticalThickness; ///< Optical thickness of material (-INF, INF)
        float glossiness;       ///< Glossiness factor
        float emissiveStrength; ///< Emissive light intensity

        Vector<UInt8,3> emissive; ///< Emissive color
        Vector<UInt8,3> diffuse;  ///< Diffuse reflection color
        Vector<UInt8,3> specular; ///< Specular reflection color

        /** Initializes a default material */
        Material() :
          opticalThickness(0.2f),
          glossiness(0.8f),
          emissiveStrength(0.0f),
          emissive(0, 0, 0),
          diffuse(255, 255, 255),
          specular(0, 0, 0)
        {
        }
    };
}

// End definition
#endif // VOX_MATERIAL_H