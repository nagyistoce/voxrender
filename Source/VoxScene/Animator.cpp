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
#include "VoxScene/Scene.h"
#include "VoxScene/Transfer.h"
#include "VoxScene/Camera.h"
#include "VoxScene/Light.h"

namespace vox {

    class Animator::Impl
    {
    public:
        Impl() : m_framerate(30) 
        {
            m_uri = "file:///" + boost::filesystem::current_path().string() + "/Temp/";
        }

        unsigned int m_framerate;
        std::list<std::pair<int,KeyFrame>> m_keys;
        ResourceId m_uri;
        String m_base;
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
std::list<std::pair<int,KeyFrame>> const& Animator::keyframes()
{
    return m_pImpl->m_keys;
}

// --------------------------------------------------------------------
//  Sets the output directory for temporary storage during rendering
// --------------------------------------------------------------------
void Animator::setTempLocation(ResourceId const& identifier, String const& baseName)
{
    m_pImpl->m_uri = identifier;
    m_pImpl->m_base = baseName;
}

// --------------------------------------------------------------------
//  Sets the output directory for temporary storage during rendering
// --------------------------------------------------------------------
ResourceId const& Animator::tempLocation()
{
    return m_pImpl->m_uri;
}

// --------------------------------------------------------------------
//  Sets the output directory for temporary storage during rendering
// --------------------------------------------------------------------
String const& Animator::baseName()
{
    return m_pImpl->m_base;
}

// --------------------------------------------------------------------
//  Performs keyframe interpolation
// --------------------------------------------------------------------
void Animator::interp(KeyFrame const& k1, KeyFrame const& k2, Scene & o, float f)
{
    k1.clone(o);

    o.camera   = k1.camera->interp(k2.camera, f);
    o.lightSet = k1.lightSet->interp(k2.lightSet, f);
    //o.transfer = k1.transfer->interp(k2.transfer, f);

    o.transferMap = TransferMap::create();
    k1.transfer->generateMap(o.transferMap);
}

// --------------------------------------------------------------------
//  Clears the internal list of keyframes
// --------------------------------------------------------------------
void Animator::clear()
{
    m_pImpl->m_keys.clear();
}

// --------------------------------------------------------------------
//  Inserts a keyframe into the scene at the specified time index
// --------------------------------------------------------------------
void Animator::addKeyframe(KeyFrame keyFrame, int frame)
{
    if (!keyFrame.isValid()) throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY,
        "Keyframe is invalid", Error_MissingData);

    auto iter = m_pImpl->m_keys.begin();
    while (iter != m_pImpl->m_keys.end() && (*iter).first < frame)
        ++iter;

    if (iter != m_pImpl->m_keys.end() && iter->first == frame)
    {
        auto old = iter; 
        ++iter;
        m_pImpl->m_keys.erase(old);
    }

    m_pImpl->m_keys.insert(iter, std::make_pair(frame, keyFrame));
}
        
// --------------------------------------------------------------------
//  Removes a keyframe at the specified frame index
// --------------------------------------------------------------------
void Animator::removeKeyframe(int frame)
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