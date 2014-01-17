/* ===========================================================================

	Project: GPU based Volume Scatter Renderer
    
	Description: Random buffer generator

    Copyright (C) 2014 Lucas Sherman

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
#include "CRandomBuffer.h"

namespace vox
{
    
// --------------------------------------------------------------------
//  Generates random content for a buffer
// --------------------------------------------------------------------
void CRandomBuffer2D::randomize()
{
    size_t bufSize = m_width * m_height;

    std::unique_ptr<unsigned int> seeds(new unsigned int[bufSize]);

    auto ptr = seeds.get();
    for (size_t i = 0; i < bufSize; i++) *ptr++ = rand();

    VOX_CUDA_CHECK(cudaMemcpy(m_pData, seeds.get(), bufSize*sizeof(unsigned int), cudaMemcpyHostToDevice));
}

} // namespace vox