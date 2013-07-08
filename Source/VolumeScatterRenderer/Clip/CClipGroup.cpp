/* ===========================================================================

    Project: Volume Scatter Renderer

    Description: Defines a grouping of other clipping elements

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
#include "CClipGroup.h"

// Include Dependencies
#include "VolumeScatterRenderer/Kernels/RenderKernel.h"

namespace vox {

// ----------------------------------------------------------------------------
//  Constructs a callback function for performing clipping for the given group
// ----------------------------------------------------------------------------
CClipGroup::CClipGroup(std::shared_ptr<PrimGroup> group) : 
  m_primGroup(group), 
  m_handle(nullptr)
{
    m_buffer.init();
}

// ----------------------------------------------------------------------------
//  Creates a Clipper object in device memory for use in clipping operations
// ----------------------------------------------------------------------------
CClipGeometry::Clipper * CClipGroup::clipper()  
{ 
    if (!m_handle)
    {
        VOX_CUDA_CHECK(cudaMalloc((void**)&m_handle, sizeof(Data)));

        std::vector<CClipGeometry::Clipper*> clippers;
        BOOST_FOREACH (auto & child, m_primGroup->children())
        {
            auto elem = CClipGeometry::create(child);

            clippers.push_back(elem->clipper());

            m_children.push_back(elem);
        }
            
        m_buffer.write(clippers);

        Data data;
        data.clipper.func = RenderKernel::getCClipGroupFunc();
        memcpy(&data.children, &m_buffer, sizeof(m_buffer));

        VOX_CUDA_CHECK(cudaMemcpy(m_handle, &data, sizeof(Data), cudaMemcpyHostToDevice));
    }

    return reinterpret_cast<Clipper*>(m_handle); 
}

} // namespace vox