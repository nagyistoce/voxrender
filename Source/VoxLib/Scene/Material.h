/* ===========================================================================

	Project: VoxRender

	Description: Data structure defining material properties of volume

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

// Begin definition
#ifndef VOX_MATERIAL_H
#define VOX_MATERIAL_H

// Include Dependencies
#include "VoxLib/Core/CudaCommon.h"
#include "VoxLib/Core/Geometry/Color.h"

// API Namespace
namespace vox
{
    /** 
     * Defines the material properties of a block of volume material.
     */
    class VOX_EXPORT Material
    {
    public:
        /** Initializes a standard default material */
        Material() :
          m_opticalThickness(0.0f),
          m_glossiness(80.0f),
          m_emissiveStrength(0.0f),
          m_emissive(0, 0, 0),
          m_diffuse(255, 255, 255),
          m_specular(0, 0, 0),
          m_dirty(true)
        {
        }

        /** Returns the optical thickness of the material () */
        float opticalThickness() const { return m_opticalThickness; }
        
        /** Sets the optical thickness of the material */
        void setOpticalThickness(float thickness)
        {
            if (m_opticalThickness != thickness)
            {
                m_opticalThickness = thickness;

                m_dirty = true;
            }
        }

        /** Returns the glossiness factor of the material */
        float glossiness() const { return m_glossiness; }

        /** Sets the glossiness factor of the material */
        void setGlossiness(float glossiness)
        {
            if (m_glossiness != glossiness)
            {
                m_glossiness = glossiness;

                m_dirty = true;
            }
        }
        
        /** Emissive light intensity */
        float emissiveStrength() const { return m_emissiveStrength; }

        /** Sets the emissive light intensity */
        void setEmissiveStrength(float intensity) { m_emissiveStrength = intensity; }

        /** Returns the emissive properties of the material */
        Vector<UInt8,3> emissive() const { return m_emissive; }

        /** Sets the emissive properties of the material */
        void setEmissive(Vector<UInt8,3> const& emissive)
        {
            if (m_emissive != emissive)
            {
                m_emissive = emissive;

                m_dirty = true;
            }
        }

        /** Returns the diffuse properties of the material */
        Vector<UInt8,3> const& diffuse() const { return m_diffuse; }

        /** Sets the diffuse properties of the material */
        void setDiffuse(Vector<UInt8,3> const& diffuse)
        {
            if (m_diffuse != diffuse)
            {
                m_diffuse = diffuse;

                m_dirty = true;
            }
        }

        /** Returns the specular properties of the material */
        Vector<UInt8,3> const& specular() const { return m_specular; }

        /** Sets the specular properties of the material */
        void setSpecular(Vector<UInt8,3> const& specular)
        {
            if (m_specular != specular)
            {
                m_specular = specular;

                m_dirty = true;
            }
        }

    private:
        //class Impl; std::unique_ptr<Impl> m_pImpl;

        float m_opticalThickness; ///< Optical thickness of material (-INF, INF)
        float m_glossiness;       ///< Glossiness factor
        float m_emissiveStrength; ///< Emissive light intensity

        Vector<UInt8,3> m_emissive;         ///< Emissive color
        Vector<UInt8,3> m_diffuse;          ///< Diffuse reflection color
        Vector<UInt8,3> m_specular;         ///< Specular reflection color

        bool m_dirty; ///< Dirty flag for interactive rendering
    };
}

// End definition
#endif // VOX_MATERIAL_H