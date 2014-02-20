/* ===========================================================================

	Project: VoxScene

	Description: Defines the Scene class used by the Renderer

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

// Begin definition
#ifndef VOX_ANIMATOR_H
#define VOX_ANIMATOR_H

// Include Dependencies
#include "VoxLib/Core/CudaCommon.h"
#include "VoxLib/Core/Types.h"

// API namespace
namespace vox
{
    class VOX_EXPORT Scene;
    typedef Scene KeyFrame;

	/** 
	 * Animation class
     *
     * This class encapsulates animation keyframes associated with a specific
     * scene instance. It also includes settings for the animation parameters.
	 */
	class VOX_EXPORT Animator
	{
    public:
        /** Constructor */
        Animator();

        /** Destructor */
        ~Animator();

        /** Adds a keyframe to the animation */
        void addKeyframe(KeyFrame keyFrame, float time);

        /** Deletes a keyframe from the animation */
        void removeKeyframe(KeyFrame keyFrame);

        /** Sets the animation framerate (in frames per second) */
        void setFramerate(unsigned int framerate);

        /** Returns the animation framerate */
        unsigned int framerate();

    private:
        class Impl;
        Impl * m_pImpl;
	};
}

// End definition
#endif // VOX_ANIMATOR_H