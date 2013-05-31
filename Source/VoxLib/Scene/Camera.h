/* ===========================================================================

	Project: VoxRender - Camera

	Description: Defines a generic 3D camera applicable to general rendering

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
#ifndef VOX_CAMERA_H
#define VOX_CAMERA_H

// Include Dependencies
#include "VoxLib/Core/CudaCommon.h"
#include "VoxLib/Core/Geometry.h"

// API namespace
namespace vox
{

class RenderController;

/** Camera Class */
class VOX_EXPORT Camera
{
public:
    /**
     * Constructor - Initalizes a scene camera to the default orientation
     *
     * The default camera position is at the origin facing along
     * the z-axis. (The coordinate system is left-handed)
     */ 
    Camera(); 

    /** Returns the current camera position */
    inline Vector3f const& position() const { return m_pos; }

    /** Returns the current camera direction vector */
    inline Vector3f const& eye() const { return m_eye; }

    /** Returns the current up direction vector of the camera */
    inline Vector3f const& up() const { return m_up; }

    /** Returns the current right direction vector of the camera */
    inline Vector3f const& right() const { return m_right; }

    /** Sets the position of the camera */
    inline void setPosition(const Vector3f &pos) { m_pos = pos; m_contextChanged = true; }

    /** Specifies a translation vector to apply to the camera position */
    inline void translate(const Vector3f &vec) { m_pos += vec; m_contextChanged = true; }

    /** Translates the camera a set distance along the x-axis */
    inline void translateX(float dist) { m_pos[0] += dist; m_contextChanged = true; }
    
    /** Translates the camera a set distance along the y-axis */
    inline void translateY(float dist) { m_pos[1] += dist; m_contextChanged = true; }
    
    /** Translates the camera a set distance along the z-axis */
    inline void translateZ(float dist) { m_pos[2] += dist; m_contextChanged = true; }
    
    /** Sets the position of the camera along the x-axis */
    inline float positionX() const { return m_pos[0]; }
    
    /** Sets the position of the camera along the y-axis */
    inline float positionY() const { return m_pos[1]; }
    
    /** Sets the position of the camera along the z-axis */
    inline float positionZ() const { return m_pos[2]; }
    
    /** Moves the camera along its current direction vector */
    inline void move(float dist) { m_pos += m_eye*dist; m_contextChanged = true; }
    
    /** Moves the camera along its current right direction vector */
    inline void moveRight(float dist) { m_pos += m_right*dist; m_contextChanged = true; }
    
    /** Moves the camera along its current right direction vector */
    inline void moveLeft(float dist) { m_pos -= m_right*dist; m_contextChanged = true; }

    /** Moves the camera along its current up direction vector */
    inline void moveUp(float dist) { m_pos += m_up*dist; m_contextChanged = true; }

    /** Moves the camera along its current up direction vector */
    inline void moveDown(float dist) { m_pos -= m_up*dist; m_contextChanged = true; }

    /** Executes a yaw rotation */
    void yaw(float radians);
    
    /** Executes a pitch rotation */
    void pitch(float radians);
    
    /** Executes a roll rotation */
    void roll(float radians);

    /** 
     * Points the camera at the specified 3D position 
     *  
     * The up parameter specifies an ideal up orientation
     * from which the camera roll will be extracted.
     */
    void lookAt(Vector3f const& pos, Vector3f const& up);

    /** Overload for lookAt which maintains the camera z-orientation */
    inline void lookAt(Vector3f const& pos) { lookAt(pos, m_up); }

    /** Sets the camera eye orientation vector */
    inline void setEye(Vector3f const& eye) { m_eye = eye; m_contextChanged = true; }

    /** Sets the camera right orientation vector */
    inline void setRight(Vector3f const& right) { m_right = right; m_contextChanged = true; }

    /** Sets the camera up orientation vector */
    inline void setUp(Vector3f const& up) { m_up = up; m_contextChanged = true; }
    
    /** Returns the camera field of view angle in radians */
    inline float fieldOfView()   const { return m_fieldOfView; }

    /** Returns the camera focal distance */
    inline float focalDistance() const { return m_focalDistance; }
    
    /** Returns the aperture size of the camera */
    inline float apertureSize()  const { return m_apertureSize; }
    
    /** Film height accessor */
    inline size_t filmHeight() const { return m_filmHeight; }

    /** Film width accessor */
    inline size_t filmWidth() const { return m_filmWidth; }

    /** Sets the camera field of view angle in radians */
    inline void setFieldOfView(float angle) { m_fieldOfView = angle; m_contextChanged = true; }

    /** Sets the camera focal distance */
    inline void setFocalDistance(float dist) { m_focalDistance = dist; m_contextChanged = true; }

    /** Sets the aperture size of the camera */
    inline void setApertureSize(float size) { m_apertureSize = size; m_contextChanged = true; }

    /** Returns the aspect ratio of the film */
    inline float aspectRatio() const { return float(m_filmWidth) / m_filmHeight; }

    /** Film height modifier */
    inline void setFilmHeight(size_t height) { m_filmHeight = height; m_filmChanged = true; }

    /** Film width modifier */
    inline void setFilmWidth(size_t width) { m_filmWidth = width; m_filmChanged = true; }

    /** Returns true if the context change flag is set */
    inline bool isDirty() const { return m_contextChanged || m_filmChanged; }

    /** Returns true if the film dimensions change flag is set */
    inline bool isFilmDirty() const { return m_filmChanged; }

private:
    friend RenderController;

    bool m_contextChanged; ///< Context change flag
    bool m_filmChanged;    ///< Film change flag

    // Camera orientation
    Vector3f m_pos;   ///< Camera position vector (mm)
    Vector3f m_eye;   ///< Forward axis direction
    Vector3f m_right; ///< Right axis direction
    Vector3f m_up;    ///< Up axis direction

	// Projection parameters
    float m_focalDistance;  ///< Focal distance (mm)
	float m_apertureSize;   ///< Aperture size  (mm)
    float m_fieldOfView;    ///< Field of view  (radians)
    float m_eyeDistance;    ///< Distance between eyes (mm)

    // Film dimensions
    size_t m_filmWidth;  ///< Film width  (pixels)
    size_t m_filmHeight; ///< Film height (pixels)
};

}

// End definition
#endif // VOX_CAMERA_H