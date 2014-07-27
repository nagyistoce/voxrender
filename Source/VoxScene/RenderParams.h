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
#include "VoxScene/Common.h"

// Include Dependencies
#include "VoxLib/Core/Common.h"
#include "VoxLib/Core/Geometry/Vector.h"

// API namespace
namespace vox
{

class RenderController;

/** 
 * Class for managing additional rendering parameters not associated with a scene element
 */
class VOXS_EXPORT RenderParams
{
public:
    /** Factory method for shared_ptr construction */
    static std::shared_ptr<RenderParams> create()
    {
        return std::shared_ptr<RenderParams>(new RenderParams());
    }
    
    /** Initializes default render parameters */
    RenderParams() :
        m_primaryStep(2.0f),
        m_shadowStep(3.0f),
        m_gradCutoff(0.0f),
        m_scatterCoefficient(0.0f),
        m_edgeEnhancement(0.0f),
        m_isDirty(true)
    {
    }
    
    /** Camera copy constructor */
    RenderParams(RenderParams & params) { params.clone(*this); }

    /** Clones the camera into an existing structure */
    void clone(RenderParams & params)
    {
        params.m_primaryStep = m_primaryStep;
        params.m_shadowStep = m_shadowStep;
        params.m_gradCutoff = m_gradCutoff;
        params.m_scatterCoefficient = m_scatterCoefficient;
        params.m_isDirty = m_isDirty;
    }

    /** Returns true if the dirty flag is set */
    bool isDirty() const { return m_isDirty; }
    
    /** Sets the dirty flag */
    void setDirty() { m_isDirty = true; }

    /** Returns the primary trace ray step size (mm) */
    float primaryStepSize() const { return m_primaryStep; }
    
    /** Returns the shadow trace ray step size (mm) */
    float shadowStepSize() const { return m_shadowStep; }
    
    /** Returns the gradient magnitude cutoff */
    float gradientCutoff() const { return m_gradCutoff; }
    
    /** Returns the scattering function coefficient */
    float scatterCoefficient() { return m_scatterCoefficient; }
    
    /** Returns the edge enhancement factor */
    float edgeEnhancement() { return m_edgeEnhancement; }

    /** Sets the gradient magnitude threshhold */
    void setGradientCutoff(float cutoff) { m_gradCutoff = cutoff; }

    /** Sets the primary trace ray step size (mm) */
    void setPrimaryStepSize(float step) { m_primaryStep = step; }
    
    /** Sets the primary trace ray step size (mm) */
    void setShadowStepSize(float step) { m_shadowStep = step; }
    
    /** Sets the scattering function coefficient */
    void setScatterCoefficient(float value) { m_scatterCoefficient = value; }

    /** Sets the edge enhancement factor */
    void setEdgeEnhancement(float enhancement) { m_edgeEnhancement = clamp(enhancement, 0.f, 100.f); }

private:
    friend RenderController;

    void setClean() { m_isDirty = false; }

    float m_primaryStep;    ///< Step size for primary volume trace
    float m_shadowStep;     ///< Step size for shadow ray trace
    float m_gradCutoff;     ///< Transition magnitude for surface/volume shading

    float m_edgeEnhancement;    ///< Edge enhancement factor
    float m_scatterCoefficient; ///< Coefficient of the scattering function

    bool m_isDirty; ///< Context change flag
};

}

// End definition
#endif // VOX_RENDER_PARAMS_H