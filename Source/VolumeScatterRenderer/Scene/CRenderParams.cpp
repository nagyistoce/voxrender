/* ===========================================================================

	Project: GPU based Volume Scatter Renderer
    
	Description: Wraps the management of GPU side render params

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
#include "CRenderParams.h"

// Include Dependencies
#include "VoxScene/RenderParams.h"

namespace vox {

// ----------------------------------------------------------------------------
//  Initializes a GPU render param structure from the scene graph element
// ----------------------------------------------------------------------------
CRenderParams::CRenderParams(std::shared_ptr<RenderParams> settings) :
    m_primaryStep( settings->primaryStepSize() ),
    m_shadowStep( settings->shadowStepSize() ),
    m_gradientCutoff( settings->gradientCutoff() ),
    m_scatterCoefficient( settings->scatterCoefficient() ),
    m_edgeEnhancement( settings->edgeEnhancement() / 200.0f ),
    m_backdropColor(1.0f, 1.0f, 1.0f)
{
}

} // namespace vox