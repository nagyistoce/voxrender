/* ===========================================================================

	Project: Transfer - Transfer Function

	Description: Transfer function applied to volume dataset

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

// Begin definition
#ifndef VOX_TRANSFER_MAP_H
#define VOX_TRANSFER_MAP_H

// Include Dependencies
#include "VoxLib/Core/CudaCommon.h"
#include "VoxLib/Core/Geometry/Image3D.h"
#include "VoxLib/Core/Geometry/Vector.h"

// API Namespace
namespace vox
{
	/** 
	 * Transfer Function Mapping 
	 *
	 * A transfer function mapping is a mapping structure used by renderers
	 * for sampling the transfer function content. 
	 */
    class VOX_EXPORT TransferMap
    {
    public:
        /** Create a new transfer function map */
        static std::shared_ptr<TransferMap> create() 
        { 
            return std::shared_ptr<TransferMap>(new TransferMap()); 
        }

        /** Destructor */
        ~TransferMap();

        /** Returns the emissive component of the transfer function map */
        Image3D<Vector<UInt8,4>> & diffuse();

        /** Returns the diffuse component of the transfer function map */
        Image3D<Vector<UInt8,4>> & specular();

        /** Returns the emissive component of the transfer function map */
        Image3D<Vector4f> & emissive();

        /** Returns the opacity component of the transfer function map */
        Image3D<float> & opacity();

        /** Returns the value range of the transfer function */
        Vector2f const& valueRange(int dim) const;

        /** Returns the value range of the transfer function */
        void setValueRange(int dim, Vector2f const& range);

        /** Locks the transfer function map for editing */
        void lock();

        /** Releases a locked transfer function mapping */
        void unlock();

        /** Sets the dirty state of the map */
        void setDirty(bool dirty);

        /** Returns the dirty state of the map */
        bool isDirty();

    private:
        /** Constructor */
        TransferMap();

        class Impl;
        Impl * m_pImpl;
    };
}

// End definition
#endif // VOX_TRANSFER_MAP_H