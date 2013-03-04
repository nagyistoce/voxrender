/* ===========================================================================

	Project: Transfer - Transfer Function

	Description: Transfer function applied to volume dataset

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
#include "Transfer.h"

// API namespace
namespace vox
{
    
// ---------------------------------------------------------
//  Removes a region from the transfer function
// ---------------------------------------------------------
void Transfer::removeRegion(Region* region)
{
    m_regions.remove(region);

    m_dirty = true;
}

// ---------------------------------------------------------
//  Adds a new region to the transfer function
// ---------------------------------------------------------
Transfer::Region* Transfer::addRegion(Transfer::Region* region)
{
    if (region == nullptr)
    {
        m_regions.push_back(new Region());
    }
    else
    {
        m_regions.push_back(region);
    }

    m_dirty = true;

    return m_regions.back();
}

} // namespace vox