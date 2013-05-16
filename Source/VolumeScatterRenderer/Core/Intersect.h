/* ===========================================================================

	Project: CUDA capable 

	Description: Data structure defining material properties of volume

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
#ifndef VSR_INTERSECT
#define VSR_INTERSECT

// Include Dependencies
#include "VoxLib/Core/CudaCommon.h"
#include "VoxLib/Core/Geometry/Vector.h"
#include "VoxLib/Core/Functors.h"

// API Namespace
namespace vox
{
    /** Defines intersection routines for basic primitives */
    class Intersect
    {
    public:
    /** 
     * Computes the intesection of a ray and a box.
     *
     * Following the execution of this function
     *
     * @param [in]      rayPos The origin of the ray
     * @param [in]      rayDir The direction vector of the ray
     * @param [in]      bmin   First point defining bounding box
     * @param [in]      bmax   Second point defining bounding box
     * @param [in][out] rayMin Indicates the minimum extent of the ray
     * @param [in][out] rayMax Indicates the maximum extent of the ray
     */
    static VOX_HOST_DEVICE bool rayBoxIntersection( 
        const Vector3f &rayPos, 
        const Vector3f &rayDir, 
        const Vector3f &bmin, 
        const Vector3f &bmax, 
	    float &rayMin, 
        float &rayMax)
    {
        Vector3f const invDir(1.0f / rayDir[0], 1.0f / rayDir[1], 1.0f / rayDir[2]);

	    Vector3f const tBMax = ( bmax - rayPos ) * invDir;
	    Vector3f const tBMin = ( bmin - rayPos ) * invDir;
    
	    Vector3f const tNear( low(tBMin[0], tBMax[0]), 
                              low(tBMin[1], tBMax[1]), 
                              low(tBMin[2], tBMax[2]) );

	    Vector3f const tFar ( high(tBMin[0], tBMax[0]), 
                              high(tBMin[1], tBMax[1]), 
                              high(tBMin[2], tBMax[2]) );
    
	    rayMin = high(rayMin, high(tNear[0], high(tNear[1], tNear[2])));
	    rayMax = low(rayMax, low(tFar[0], low(tFar[1], tFar[2])));

        return rayMin > rayMax;
    }

    private:
        Intersect() {}
    };
}

// End definition
#endif // VSR_INTERSECT