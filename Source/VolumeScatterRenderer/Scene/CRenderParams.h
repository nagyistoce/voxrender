/* ===========================================================================

	Project: GPU based Volume Scatter Renderer
    
	Description: Wraps the management of GPU side render params

    Copyright (C) 2013-2014 Lucas Sherman

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
#ifndef VSR_CRENDER_PARAMS_H
#define VSR_CRENDER_PARAMS_H

// Common Library Header
#include "VolumeScatterRenderer/Core/Common.h"

// VoxLib Dependencies
#include "VoxLib/Core/Geometry.h"

// API namespace
namespace vox
{

class RenderParams;

/** Rendering Volume Class */
class CRenderParams
{
public:
    VOX_DEVICE CRenderParams() { }

    /** Sets the parameter structures data content */
    VOX_HOST CRenderParams(std::shared_ptr<RenderParams> settings);

    VOX_HOST_DEVICE float        scatterCoefficient()   const { return m_scatterCoefficient; }
    VOX_HOST_DEVICE float        primaryStepSize()      const { return m_primaryStep; }
    VOX_HOST_DEVICE float        shadowStepSize()       const { return m_shadowStep; }
    VOX_HOST_DEVICE float        occludeStepSize()      const { return m_occludeStep; }
    VOX_HOST_DEVICE unsigned int occludeSamples()       const { return m_occludeSamples; }
    VOX_HOST_DEVICE float        gradientCutoff()       const { return m_gradientCutoff; }

private:
    Vector3f m_backdropColor; ///< Background color/radiance (non-reflecting)

    float m_primaryStep;    ///< Step size for primary volume trace
    float m_shadowStep;     ///< Step size for shadow ray trace
    float m_occludeStep;    ///< Step size for ambient occlusion

    float m_scatterCoefficient; ///< Scattering coefficient for volume shading

    float m_gradientCutoff; ///< Cutoff for surface based shading 

    unsigned int m_occludeSamples; ///< Number of ambient occlusion samples
};

}

// End definition
#endif // VSR_CRENDER_PARAMS_H