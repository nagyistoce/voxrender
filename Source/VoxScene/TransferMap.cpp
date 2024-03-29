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

// Include Header
#include "TransferMap.h"

// Include Dependencies
#include "VoxLib/Core/Common.h"
#include "VoxLib/Error/Error.h"

// API namespace
namespace vox 
{

// Impl structure for TransferMap
class TransferMap::Impl
{
public:
    Image3D<Vector<UInt8,4>> diffuse;  ///< Diffuse transfer mapping [RGBX]
    Image3D<Vector<UInt8,4>> specular; ///< Specular transfer mapping [Reflectance + Roughness]
    Image3D<Vector4f>        emissive; ///< Emissive transfer mapping
    Image3D<float>           opacity;  ///< Absorption coefficient
    Vector2f                 range[3]; ///< The value range for the transfer function (as a subset of the data set range)
};

TransferMap::TransferMap()
{
    m_pImpl = new Impl();
}

TransferMap::~TransferMap()
{
    delete m_pImpl;
}

void TransferMap::setValueRange(int dim, Vector2f const& range)
{
    if (dim < 0 || dim > 3) throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY, "Invalid dim", Error_Range);

    m_pImpl->range[dim] = range;
}

Vector2f const& TransferMap::valueRange(int dim) const
{
    if (dim < 0 || dim > 3) throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY, "Invalid dim", Error_Range);

    return m_pImpl->range[dim];
}

Image3D<Vector<UInt8,4>> & TransferMap::diffuse() 
{
    return m_pImpl->diffuse;
}

Image3D<Vector4f> & TransferMap::emissive()
{
    return m_pImpl->emissive;
}

Image3D<Vector<UInt8,4>> & TransferMap::specular()
{
    return m_pImpl->specular;
}

Image3D<float> & TransferMap::opacity()
{
    return m_pImpl->opacity;
}

}
