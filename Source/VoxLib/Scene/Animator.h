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
        /** Creates a new animator */
        static std::shared_ptr<Animator> create();

        /** Destructor */
        ~Animator();

        /** Returns a list of the keyframes */
        std::list<std::pair<unsigned int,KeyFrame>> const& keyframes();

        /** Generates an interpolated keyframe for rendering: k1*f + k2*(1-f) */
        void lerp(KeyFrame const& k1, KeyFrame const& k2, Scene & o, float f);

        /** Adds a keyframe to the animation */
        void addKeyframe(KeyFrame keyFrame, unsigned int frame);

        /** Deletes a keyframe from the animation */
        void removeKeyframe(unsigned int frame);

        /** Sets the animation framerate (in frames per second) */
        void setFramerate(unsigned int framerate);

        /** Returns the animation framerate */
        unsigned int framerate();

    private:
        /** Constructor */
        Animator();

        class Impl;
        Impl * m_pImpl;
	};
}

// End definition
#endif // VOX_ANIMATOR_H