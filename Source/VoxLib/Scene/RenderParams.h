/* ===========================================================================

	Project: VoxRender - Render Parameters

	Description: Encapsulates various render parameters for a scene

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
#ifndef VOX_RENDER_PARAMS_H
#define VOX_RENDER_PARAMS_H

// Include Dependencies
#include "VoxLib/Core/CudaCommon.h"
#include "VoxLib/Core/Geometry/Vector.h"

// API namespace
namespace vox
{

class RenderController;

/** 
 * Class for managing additional rendering parameters not associated with a scene element
 */
class VOX_EXPORT RenderParams
{
public:
    static std::shared_ptr<RenderParams> create()
    {
        return std::shared_ptr<RenderParams>(new RenderParams());
    }

    /** Returns true if the context change flag is set */
    inline bool isDirty() const { return m_contextChanged; }
    
    /** Returns the primary trace ray step size (mm) */
    float primaryStepSize() const { return m_primaryStep; }
    
    /** Returns the shadow trace ray step size (mm) */
    float shadowStepSize() const { return m_shadowStep; }
    
    /** Returns the ambient occlusion trace ray step size (mm) */
    float occludeStepSize() const { return m_occludeStep; }

    /** Returns the number of ambient occlusion rays cast */
    unsigned int occludeSamples() const { return m_occludeSamples; }

    /** Returns the gradient magnitude cutoff */
    float gradientCutoff() const { return m_gradCutoff; }
    
    /** Returns the scattering function coefficient */
    float scatterCoefficient() { return m_scatterCoefficient; }

    /** Sets the gradient magnitude threshhold */
    void setGradientCutoff(float cutoff) { m_gradCutoff = cutoff; }

    /** Sets the primary trace ray step size (mm) */
    void setPrimaryStepSize(float step) { m_primaryStep = step; m_contextChanged = true; }
    
    /** Sets the primary trace ray step size (mm) */
    void setShadowStepSize(float step) { m_shadowStep = step; m_contextChanged = true; }
    
    /** Sets the primary trace ray step size (mm) */
    void setOccludeStepSize(float step) { m_occludeStep = step; m_contextChanged = true; }

    /** Sets the number of ambient occlusion sample rays cast */
    void setOccludeSamples(unsigned int samples) { m_occludeSamples = samples; m_contextChanged = true; }

    /** Sets the scattering function coefficient */
    void setScatterCoefficient(float value) { m_scatterCoefficient = value; m_contextChanged = true; }

private:
    /** Initializes default render parameters */
    RenderParams() :
        m_primaryStep(2.0f),
        m_shadowStep(3.0f),
        m_occludeStep(1.0f),
        m_occludeSamples(0u),
        m_gradCutoff(0.2f),
        m_scatterCoefficient(0.0f),
        m_contextChanged(true)
    {
    }

    friend RenderController;

    float m_primaryStep;    ///< Step size for primary volume trace
    float m_shadowStep;     ///< Step size for shadow ray trace
    float m_occludeStep;    ///< Step size for ambient occlusion
    float m_gradCutoff;     ///< Transition magnitude for surface/volume shading

    float m_scatterCoefficient; ///< Coefficient of the scattering function

    unsigned int m_occludeSamples; ///< Number of ambient occlusion samples

    bool m_contextChanged; ///< Context change flag
};

}

// End definition
#endif // VOX_RENDER_PARAMS_H