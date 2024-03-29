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
#include "VoxScene/Common.h"
#include "VoxLib/Core/CudaCommon.h"
#include "VoxLib/Core/Types.h"
#include "VoxLib/IO/ResourceId.h"

// API namespace
namespace vox
{
    class VOXS_EXPORT Scene;
    typedef Scene KeyFrame;

	/** 
	 * Animation class
     *
     * This class encapsulates animation keyframes associated with a specific
     * scene instance. It also includes settings for the animation parameters.
	 */
	class VOXS_EXPORT Animator
	{
    public:
        /** Creates a new animator */
        static std::shared_ptr<Animator> create();

        /** Destructor */
        ~Animator();

        /** Returns a list of the keyframes */
        std::list<std::pair<int,std::shared_ptr<KeyFrame>>> const& keyframes();

        /** Clears the list of keyframe data */
        void clear();

        /** Sets the base URI for the temporary storage of frame information */
        void setTempLocation(ResourceId const& identifier, String const& baseName);
        
        /** Returns the base URI for temporary frame storage during rendering */
        ResourceId const& tempLocation();

        /** Returns the base filename for temporary frames produced during rendering */
        String const& baseName();

        /** Sets the video output URI */
        void setOutputUri(ResourceId const& output);

        /** Returns the current output URI for the video */
        ResourceId const& outputUri();

        /** Generates an interpolated keyframe for rendering: k1*f + k2*(1-f) */
        std::shared_ptr<Scene> interp(
            std::shared_ptr<KeyFrame> k1, 
            std::shared_ptr<KeyFrame> k2,  
            float f, 
            std::shared_ptr<Scene> o = nullptr);

        /** Adds a keyframe to the animation */
        void addKeyframe(std::shared_ptr<KeyFrame> keyFrame, int frame, bool suppress = false);

        /** Deletes a keyframe from the animation */
        void removeKeyframe(int frame, bool suppress = false);

        /** Sets the animation framerate (in frames per second) */
        void setFramerate(unsigned int framerate);

        /** Returns the animation framerate */
        unsigned int framerate();
        
        /** Sets the callback event for adding a light */
        void onAdd(std::function<void(int, std::shared_ptr<KeyFrame>, bool)> callback);

        /** Sets the callback event for removing a light */
        void onRemove(std::function<void(int, std::shared_ptr<KeyFrame>, bool)> callback);

    private:
        /** Constructor */
        Animator();

        class Impl;
        Impl * m_pImpl;
	};
}

// End definition
#endif // VOX_ANIMATOR_H