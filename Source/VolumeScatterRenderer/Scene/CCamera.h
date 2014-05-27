/* ===========================================================================

	Project: VoxRender - Device Side Camera

	Description: Defines a 3D Camera for use in rendering

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

// Begin definition
#ifndef VSR_CCAMERA_H
#define VSR_CCAMERA_H

// Common Library Header
#include "VolumeScatterRenderer/Core/Common.h"

// VoxLib Dependencies
#include "VoxLib/Core/Geometry.h"

// API namespace
namespace vox
{

class Camera;

/** Rendering Camera Class */
class CCamera
{
public:
    VOX_HOST_DEVICE CCamera() { }

    /** Constructs a device camera for the specified scene */
    VOX_HOST CCamera(std::shared_ptr<Camera> const& camera);

    /** Returns the current camera position */
    VOX_HOST_DEVICE inline Vector3f const& position() const { return m_pos; }

    /** Returns the current camera direction vector */
    VOX_HOST_DEVICE inline Vector3f const& eye() const { return m_eye; }

    /** Returns the current up direction vector of the camera */
    VOX_HOST_DEVICE inline Vector3f const& up() const { return m_up; }

    /** Returns the current right direction vector of the camera */
    VOX_HOST_DEVICE inline Vector3f const& right() const { return m_right; }

    /** Returns the camera focal distance */
    VOX_HOST_DEVICE inline float focalDistance() const { return m_focalDistance; }
    
    /** Returns the aperture size of the camera */
    VOX_HOST_DEVICE inline float apertureSize()  const { return m_apertureSize; }

    /** Generates a sampling ray for the camera */
    VOX_HOST_DEVICE inline Ray3f generateRay(
        Vector2f const& screenCoords, 
        Vector2f const& apertureRnd,
        int isLeftEye) const
    {
        float screenX = m_screenUpperLeft[0] + m_screenPerPixel[0] * screenCoords[0];
        float screenY = m_screenUpperLeft[1] + m_screenPerPixel[1] * screenCoords[1];

        auto offset = m_right * m_eyeDistance;
        if (isLeftEye) offset = - offset;
        Ray3f ray(m_pos + offset, m_eye - offset / m_focalDistance + (m_right * screenX) - (m_up * screenY));

        ray.dir.normalize();
        ray.min = 0.0f;
        ray.max = 100000.0f;

        if (m_apertureSize != 0.0f)
        {
            Vector2f lensUV = apertureRnd * m_apertureSize;
            Vector3f LI     = m_right * lensUV[0] + m_up * lensUV[1];

            ray.pos += LI;
            ray.dir *= m_focalDistance; 
            ray.dir -= LI;
            ray.dir.normalize();
        }

        return ray;
    }

private:
    Vector2f m_screenUpperLeft; ///< Base screen position (UL Corner)
    Vector2f m_screenPerPixel;  ///< Screen distance / pixel
    
    Vector3f m_pos;         ///< Camera position vector
    Vector3f m_eye;         ///< Forward axis direction
    Vector3f m_right;       ///< Right axis direction
    Vector3f m_up;          ///< Up axis direction
 
    float m_focalDistance;  ///< Focal distance (mm)
	float m_apertureSize;   ///< Aperture size  (mm)
    float m_eyeDistance;    ///< Eye distance (mm)
};

}

// End definition
#endif // VSR_CCAMERA_H