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

// Include Header
#include "Animator.h"

// Include Dependencies
#include "VoxLib/Scene/Scene.h"
#include "VoxLib/Scene/Transfer.h"
#include "VoxLib/Scene/Camera.h"

namespace vox {

    class Animator::Impl
    {
    public:
        Impl() : m_framerate(30) { }

        unsigned int m_framerate;
        std::list<std::pair<unsigned int,KeyFrame>> m_keys;
    };
    
// --------------------------------------------------------------------
//  Creates a new animator object
// --------------------------------------------------------------------
std::shared_ptr<Animator> Animator::create()
{
    return std::shared_ptr<Animator>(new Animator());
}

// --------------------------------------------------------------------
//  Constructor
// --------------------------------------------------------------------
Animator::Animator() : m_pImpl(new Impl()) 
{
}

// --------------------------------------------------------------------
//  Destructor
// --------------------------------------------------------------------
Animator::~Animator() 
{ 
    delete m_pImpl; 
}

// --------------------------------------------------------------------
//  Returns the internal map of keyframes 
// --------------------------------------------------------------------
std::list<std::pair<unsigned int,KeyFrame>> const& Animator::keyframes()
{
    return m_pImpl->m_keys;
}

// --------------------------------------------------------------------
//  Performs keyframe interpolation
// --------------------------------------------------------------------
void Animator::lerp(KeyFrame const& k1, KeyFrame const& k2, Scene & o, float f)
{
    k1.clone(o);
    o.transferMap = TransferMap::create();
    k1.transfer->generateMap(o.transferMap);
}

// --------------------------------------------------------------------
//  Inserts a keyframe into the scene at the specified time index
// --------------------------------------------------------------------
void Animator::addKeyframe(KeyFrame keyFrame, unsigned int frame)
{
    auto iter = m_pImpl->m_keys.begin();
    while (iter != m_pImpl->m_keys.end() && (*iter).first < frame)
        ++iter;

    m_pImpl->m_keys.insert(iter, std::make_pair(frame, keyFrame));
}
        
// --------------------------------------------------------------------
//  Removes a keyframe at the specified frame index
// --------------------------------------------------------------------
void Animator::removeKeyframe(unsigned int frame)
{
    for (auto iter = m_pImpl->m_keys.begin(); iter != m_pImpl->m_keys.end(); ++iter)
    if ((*iter).first == frame)
    {
        m_pImpl->m_keys.erase(iter);
        return;
    }
}

// --------------------------------------------------------------------
//  Sets the framerate
// --------------------------------------------------------------------
void Animator::setFramerate(unsigned int framerate) 
{ 
    m_pImpl->m_framerate = framerate; 
}

// --------------------------------------------------------------------
//  Returns the framerate
// --------------------------------------------------------------------
unsigned int Animator::framerate() 
{ 
    return m_pImpl->m_framerate; 
}

} // namespace vox