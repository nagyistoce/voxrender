/* ===========================================================================

	Project: VoxRender - Camera
    
	Description: Defines a 3D Camera for controlling the imaging parameters.

    Copyright (C) 2012 Lucas Sherman

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
#include "Camera.h"

#include <math.h>

// API namespace
namespace vox
{

// --------------------------------------------------------------------
//  Initialize the camera settings to default values
// --------------------------------------------------------------------
Camera::Camera() :
    m_pos   (0.0f, 0.0f, 0.0f),
    m_eye   (0.0f, 0.0f, 1.0f),
    m_right (1.0f, 0.0f, 0.0f),
    m_up    (0.0f, 1.0f, 0.0f),
    m_focalDistance  (0.0f),
    m_apertureSize   (0.0f),
    m_fieldOfView    (M_PI / 6),
    m_eyeDistance    (0.0f),
    m_filmWidth      (512),
    m_filmHeight     (512),
    m_isFilmDirty    (false),
    m_isStereo       (false)
{ 
}

// --------------------------------------------------------------------
//  Destructor
// --------------------------------------------------------------------
Camera::~Camera()
{
}

// --------------------------------------------------------------------
//  Clones the camera into an existing structure
// --------------------------------------------------------------------
void Camera::clone(Camera & camera)
{
    camera.m_id = m_id;
    camera.m_isDirty = m_isDirty;
    camera.m_isFilmDirty = m_isFilmDirty;
    camera.m_pos = m_pos;
    camera.m_eye = m_eye;
    camera.m_right = m_right;
    camera.m_up = m_up;
    camera.m_focalDistance = m_focalDistance;
    camera.m_apertureSize = m_apertureSize;
    camera.m_fieldOfView = m_fieldOfView;
    camera.m_eyeDistance = m_eyeDistance;
    camera.m_filmWidth = m_filmWidth;
    camera.m_filmHeight = m_filmHeight;
}

// --------------------------------------------------------------------
//  Interpolates between camera keyframes
// --------------------------------------------------------------------
std::shared_ptr<Camera> Camera::interp(std::shared_ptr<Camera> k2, float f)
{
    std::shared_ptr<Camera> result = Camera::create();

    auto key1 = this;
    auto key2 = k2.get();

#define VOX_LERP(ATTR) result->ATTR = key1->ATTR*(1-f) + key2->ATTR*f;

    VOX_LERP(m_pos);
    VOX_LERP(m_apertureSize);
    VOX_LERP(m_fieldOfView);
    VOX_LERP(m_focalDistance);

    VOX_LERP(m_eye); result->m_eye.normalize(); // :TODO:
    VOX_LERP(m_up); result->m_up.normalize();
    result->setRight(Vector3f::cross(result->m_eye, result->m_up).normalized());

    result->m_filmWidth  = m_filmWidth;
    result->m_filmHeight = m_filmHeight;
        
#undef VOX_LERP

    return result;
}

// --------------------------------------------------------------------
//  Executes a yaw movement of the camera
// --------------------------------------------------------------------
void Camera::yaw(float radians)
{
    float cosA = cos(radians);
    float sinA = sin(radians);

    auto neye   = m_eye * cosA - m_right * sinA;
    auto nright = m_eye * sinA + m_right * cosA;

    m_eye   = neye.normalized();
    m_right = nright.normalized();
}

// --------------------------------------------------------------------
//  Executes a pitch movement of the camera
// --------------------------------------------------------------------
void Camera::pitch(float radians)
{
    float cosA = cos(radians);
    float sinA = sin(radians);

    auto neye = m_eye * cosA - m_up * sinA;
    auto nup  = m_eye * sinA + m_up * cosA;

    m_eye = neye.normalized();
    m_up  = nup.normalized();
}

// --------------------------------------------------------------------
//  Executes a roll movement of the camera
// --------------------------------------------------------------------
void Camera::roll(float radians)
{
    float cosA = cos(radians);
    float sinA = sin(radians);

    auto nup    = m_up * cosA - m_right * sinA;
    auto nright = m_up * sinA + m_right * cosA;

    m_right = nright.normalized();
    m_up    = nup.normalized();
}

// --------------------------------------------------------------------
//  Points the camera at a specified 3D coordinate
// --------------------------------------------------------------------
void Camera::lookAt(Vector3f const& position, Vector3f const& up)
{
    auto eyeish = (position-m_pos);
    m_eye = (position-m_pos).normalized();

    m_right = Vector3f::cross(m_eye, up).normalized();
    m_up    = Vector3f::cross(m_right, m_eye).normalized();
}

// --------------------------------------------------------------------
//  Generates a camera ray for the specified film coordinates
// --------------------------------------------------------------------
Ray3f Camera::projectRay(Vector2f const& screenCoords)
{
    // Precompute screen sampling parameters
    float wo = tanf(m_fieldOfView / 2.0f);
    float ho = wo * m_filmHeight / m_filmWidth;
    auto screenUpperLeft = Vector2f(-wo, -ho);
    auto screenPerPixel  = Vector2f(wo / m_filmWidth, ho / m_filmHeight) * 2.0f;

    float screenX = screenUpperLeft[0] + screenPerPixel[0] * screenCoords[0];
    float screenY = screenUpperLeft[1] + screenPerPixel[1] * screenCoords[1];

    Ray3f ray(m_pos, m_eye + (m_right * screenX) - (m_up * screenY));

    ray.dir.normalize();
    ray.min = 0.0f;
    ray.max = std::numeric_limits<float>::infinity();

    return ray;
}

}