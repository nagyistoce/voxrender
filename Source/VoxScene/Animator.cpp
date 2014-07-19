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
#include "VoxScene/Volume.h"
#include "VoxScene/PrimGroup.h"

namespace vox {

    class Animator::Impl
    {
    public:
        Impl() : m_framerate(30) 
        {
            m_uri = "file:///" + boost::filesystem::current_path().string() + "/Temp/";
        }

        std::function<void(int, std::shared_ptr<KeyFrame>, bool)> addCallback;
        std::function<void(int, std::shared_ptr<KeyFrame>, bool)> remCallback;

        unsigned int m_framerate;
        std::list<std::pair<int,std::shared_ptr<KeyFrame>>> m_keys;
        ResourceId m_uri;
        ResourceId m_videoUri;
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
std::list<std::pair<int,std::shared_ptr<KeyFrame>>> const& Animator::keyframes()
{
    return m_pImpl->m_keys;
}

// --------------------------------------------------------------------
//  Sets the output directory for the final video
// --------------------------------------------------------------------
void Animator::setOutputUri(ResourceId const& identifier)
{
    m_pImpl->m_videoUri = identifier;
}

// --------------------------------------------------------------------
//  Returns the output directory for the final video
// --------------------------------------------------------------------
ResourceId const& Animator::outputUri()
{
    return m_pImpl->m_videoUri;
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
std::shared_ptr<Scene> Animator::interp(
    std::shared_ptr<KeyFrame> k1, 
    std::shared_ptr<KeyFrame> k2, 
    float f, 
    std::shared_ptr<Scene> o)
{
    if (!k1 || !k2) throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY,
        "Cannot interpolate between null keyframes", Error_MissingData);

    auto scene = o ? o : Scene::create();

    scene->clipGeometry = std::dynamic_pointer_cast<PrimGroup>(
        k1->clipGeometry->interp(k2->clipGeometry, f));

    scene->camera       = k1->camera->interp(k2->camera, f);
    scene->lightSet     = k1->lightSet->interp(k2->lightSet, f);
    scene->volume       = k1->volume->interp(k2->volume, f);
    scene->parameters   = k1->parameters;

    if (!k1->transferMap)
    {
        if (k1->transfer)
        {
            scene->transfer = k1->transfer->interp(k2->transfer, f);
            scene->transferMap = TransferMap::create();
            scene->transfer->generateMap(o->transferMap);
        }
        else
        {
            scene->transfer = nullptr;
            scene->transferMap = nullptr;
        }
    }
    else
    {
        scene->transferMap = k1->transferMap;
    }

    return scene;
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
void Animator::addKeyframe(std::shared_ptr<KeyFrame> keyFrame, int frame, bool suppress)
{
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

    if (m_pImpl->addCallback) m_pImpl->addCallback(frame, keyFrame, suppress);
}
        
// --------------------------------------------------------------------
//  Removes a keyframe at the specified frame index
// --------------------------------------------------------------------
void Animator::removeKeyframe(int frame, bool suppress)
{
    for (auto iter = m_pImpl->m_keys.begin(); iter != m_pImpl->m_keys.end(); ++iter)
    if ((*iter).first == frame)
    {
        auto key = iter->second;
        m_pImpl->m_keys.erase(iter);
        if (m_pImpl->remCallback) m_pImpl->remCallback(frame, key, suppress);
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

// ----------------------------------------------------------------------------
//  Callback event modifier functions
// ----------------------------------------------------------------------------
void Animator::onAdd(std::function<void(int, std::shared_ptr<KeyFrame>, bool)> callback)    
{ 
    m_pImpl->addCallback = callback; 
}
void Animator::onRemove(std::function<void(int, std::shared_ptr<KeyFrame>, bool)> callback) 
{ 
    m_pImpl->remCallback = callback; 
}

} // namespace vox