/* ===========================================================================

	Project: VoxRender - Color

	Defines a class for handling color information in various common formats.

    Copyright (C) 2013 Lucas Sherman

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
#ifndef VOX_COLOR_H
#define VOX_COLOR_H

// Include Dependencies
#include "VoxLib/Core/CudaCommon.h"
#include "VoxLib/Core/Geometry/Vector.h"
#include "VoxLib/Core/Types.h"

// API namespace
namespace vox
{
	/** Low Dynamic Range RGB color information */
	struct VOX_EXPORT ColorRgbLdr
	{
    public:
        VOX_HOST_DEVICE ColorRgbLdr() { }

        /** */
        VOX_HOST_DEVICE ColorRgbLdr(UInt8 r, UInt8 g, UInt8 b) 
        { 
        }
        
        /** */
        VOX_HOST_DEVICE ColorRgbLdr(Vector<UInt8,3> rgb) 
        { 
        }
        
        /** Computes the sum of the color components */
        VOX_HOST_DEVICE void operator+=(ColorRgbLdr const& rhs)
        {
            r += rhs.r; g += rhs.g; b += rhs.b;
        }
        
        /** Computes the sum of the color components */
        VOX_HOST_DEVICE ColorRgbLdr operator+(ColorRgbLdr const& rhs) const
        {
            return ColorRgbLdr(r+rhs.r, g+rhs.g, b+rhs.b);
        }

        /** Color equality comparison operator */
        VOX_HOST_DEVICE bool operator==(ColorRgbLdr const& rhs)
        {
            return (r==rhs.r) && (g==rhs.g) && (b==rhs.b);
        }

        UInt8 r; ///< Red component
        UInt8 g; ///< Green component
        UInt8 b; ///< Blue component
	};
	
    /** Low Dynamic Range RGBA color information */
	struct VOX_EXPORT ColorRgbaLdr 
	{
        VOX_HOST_DEVICE ColorRgbaLdr() { } 

        /** Initialization constructor */
        VOX_HOST_DEVICE ColorRgbaLdr(
            UInt8 ir, UInt8 ig, UInt8 ib, UInt8 ia = 0xF) :
          r(ir), g(ig), b(ib), a(ia)
        {
        }
          
        /** Color equality comparison operator */
        VOX_HOST_DEVICE bool operator==(ColorRgbaLdr const& rhs)
        {
            return (r==rhs.r) && (g==rhs.g) && (b==rhs.b) && (a==rhs.a);
        }

        UInt8 b; ///< Blue component
        UInt8 g; ///< Green component
        UInt8 r; ///< Red component
        UInt8 a; ///< Alpha component
	};

    /** High Dynamic Range CIE LAB Color Information */
    struct VOX_EXPORT ColorLabHdr
    {
    public:
        VOX_HOST_DEVICE ColorLabHdr() { }

        /** Initialization constructor for valid components */
        VOX_HOST_DEVICE ColorLabHdr(float lc, float ac, float bc) :
            l(lc), a(ac), b(bc)
        { 
        }

        /** Distance metric on LAB color space */
        VOX_HOST_DEVICE static float distance(
            ColorLabHdr const& lhs, 
            ColorLabHdr const& rhs)
        {
            return sqrtf(lhs.l*rhs.l +
                         lhs.a*rhs.a +
                         lhs.b*rhs.b);
        }

        /** Distance metric on LAB color space */
        VOX_HOST_DEVICE float distance(ColorLabHdr const& rhs)
        {
            ColorLabHdr const& lhs = *this;
            return distance(lhs, rhs);
        }
        
        /** Color equality comparison operator */
        VOX_HOST_DEVICE bool operator==(ColorLabHdr const& rhs) const
        {
            return (l==rhs.l) && (a==rhs.a) && (b==rhs.b);
        }

        /** Color inequality comparison operator */
        VOX_HOST_DEVICE bool operator!=(ColorLabHdr const& rhs) const
        {
            return (l!=rhs.l) && (a!=rhs.a) && (b!=rhs.b);
        }

    public:
        float l; ///< Lightness component
        float a; ///< Chromaticity component 1
        float b; ///< Chromaticity component 2
    };

    /** High Dynamic Range CIE LAB Color Information */
    struct VOX_EXPORT ColorLabxHdr : public ColorLabHdr
    {
        VOX_HOST_DEVICE ColorLabxHdr() { }

        /** Initialization constructor for valid components */
        VOX_HOST_DEVICE ColorLabxHdr(float lc, float ac, float bc) :
            ColorLabHdr(lc, ac, bc)
        { 
        }

        float x; ///< Alignment component
    };

    /** High Dynamic Range RGB color information */
    struct VOX_EXPORT ColorRgbHdr
    {
        VOX_HOST_DEVICE ColorRgbHdr() { } 

        /** Returns the distance between two colors */
        float distance(ColorRgbHdr const& rhs) const
        {
            return 0.f;
        }
        
        /** Color equality comparison operator */
        VOX_HOST_DEVICE bool operator==(ColorRgbHdr const& rhs)
        {
            return (r==rhs.r) && (g==rhs.g) && (b==rhs.b);
        }

        float r; ///< Red component
        float g; ///< Green component
        float b; ///< Blue component
    };

    /** High Dynamic Range RGBA color information */
    struct VOX_EXPORT ColorRgbaHdr
    {
        VOX_HOST_DEVICE ColorRgbaHdr() { } 
        
        float a; ///< Alpha component
        float r; ///< Red component
        float g; ///< Green component
        float b; ///< Blue component
    };

    /** Converts an LAB format color to a CIE-XYZ format equivalent */
    /*
    void labToXyz(ColorLabHdr const& lab, ColorXyzHdr & xyz
        ColorXyz & white = Vector3f(109.85f, 100.0f, 35.58f))
    {
        // http://www.easyrgb.com/index.php?X=MATH

        static const float e = 0.008856f;
        static const float k = 903.3f;

        float fy = (lab.l + 16.0f) / 116.0f;
        xyz.y = (xyz.l > e*k) ? (fy*fy*fy) : (lab.l / k);
        xyz.y *= white.y;

        float fx   = fy + lab.a / 500.0f;
        float fx_3 = fx * fx * fx;
        xyz.x = (fx_3 > e) ? fx_3 : (116.0f * fx - 16.0f)/k;
        xyz.x *= white.x;

        float fz   = fy - lab.b / 200.0f;
        float fz_3 = fz * fz * fz;
        xyz.z = (fz_3 > e) ? fz_3 : (116.0f * fz - 16.0f)/k;
        xyz.z *= white.z;
    }
    */
}

// End Definition
#endif // VOX_COLOR_H