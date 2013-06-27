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
    m_fieldOfView    (60.0f),
    m_eyeDistance    (0.0f),
    m_contextChanged (false)
{ 
}

// --------------------------------------------------------------------
//  Destructor
// --------------------------------------------------------------------
Camera::~Camera()
{
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

    m_contextChanged = true;
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

    m_contextChanged = true;
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

    m_contextChanged = true;
}

// --------------------------------------------------------------------
//  Points the camera at a specified 3D coordinate
// --------------------------------------------------------------------
void Camera::lookAt(Vector3f const& position, Vector3f const& up)
{
    m_eye = (m_pos-position).normalized();

    m_contextChanged = true;
}

}