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
#include "VoxLib/Core/CudaCommon.h"
#include "VoxLib/Core/Geometry/Color.h"

// API Namespace
namespace vox
{
    class VOX_EXPORT Node;

    /** Defines the material properties of a volume. */
    class VOX_EXPORT Material
    {
    public:
        /** Initializes a standard default material */
        Material();

        /** Because some people (VS 2010 - hint, hint) don't support unique_ptr */
        ~Material();

        /** Returns the optical thickness of the material () */
        float opticalThickness() const;
        
        /** Sets the optical thickness of the material */
        void setOpticalThickness(float thickness);

        /** Returns the glossiness factor of the material */
        float glossiness() const;

        /** Sets the glossiness factor of the material */
        void setGlossiness(float glossiness);
        
        /** Emissive light intensity */
        float emissiveStrength() const;

        /** Sets the emissive light intensity */
        void setEmissiveStrength(float intensity);

        /** Returns the emissive properties of the material */
        Vector<UInt8,3> emissive() const;

        /** Sets the emissive properties of the material */
        void setEmissive(Vector<UInt8,3> const& emissive);

        /** Returns the diffuse properties of the material */
        Vector<UInt8,3> const& diffuse() const;

        /** Sets the diffuse properties of the material */
        void setDiffuse(Vector<UInt8,3> const& diffuse);

        /** Returns the specular properties of the material */
        Vector<UInt8,3> const& specular() const;

        /** Sets the specular properties of the material */
        void setSpecular(Vector<UInt8,3> const& specular);

        /** Sets the dirty state of the material */
        void setDirty(bool dirty = true);

        /** Returns true if the dirt flag is set */
        bool isDirty() { return m_isDirty; }

    private:
        class Impl; Impl * m_pImpl;

        friend Node;

        /** Registers a user node for this material */
        void addNode(std::shared_ptr<Node> node);

        /** Deregisters a user node for this material */
        void removeNode(std::shared_ptr<Node> node);

        bool m_isDirty; ///< Dirty flag for interactive rendering
    };
}

// End definition
#endif // VOX_MATERIAL_H