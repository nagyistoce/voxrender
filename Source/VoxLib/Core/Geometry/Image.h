/* ===========================================================================

	Project: VoxRender - Image class

	Description: Defines the image class for rendering operations

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
#ifndef VOX_IMAGE_H
#define VOX_IMAGE_H

// Include Dependencies
#include "VoxLib/Core/CudaCommon.h"
#include "VoxLib/Core/Types.h"

// API namespace
namespace vox
{
	/** Rendering frame buffer class */
    template<typename T> class Image
	{
	public:
        /** Constructs and image of the specified dimensions */
		Image(size_t width, size_t height) :
            m_width(width), m_height(height), 
            m_stride(sizeof(T)*width),
            m_buffer(new T[width*height]) 
        { 
        }

        /** Constructs and image of the specified dimensions */
		Image(size_t width, size_t height, size_t stride) :
            m_width(width), m_height(height), m_stride(stride),
            m_buffer(reinterpret_cast<T*>(new UInt8[stride*height])) 
        { 
        }
            
        /** Resizes the image to the specified dimensions */
        void resize(size_t width, size_t height)
        {
            resize(width, height, width*sizeof(T));
        }

        /** Resizes the image to the specified dimensions */
        void resize(size_t width, size_t height, size_t stride)
        {
            m_width = width; m_height = height; m_stride = stride;

            m_buffer = std::unique_ptr<T[]>(reinterpret_cast<T*>(new UInt8[stride*height]));
        }

        /** Image width accessor */
        size_t width() const { return m_width; }

        /** Image height accessor */
        size_t height() const { return m_height; }

        /** Image stride accessor */
        size_t stride() const { return m_stride; }

        /** Image buffer accessor for const data access */
        T const* data() const { return m_buffer.get(); }

        /** Image buffer accessor for mutable data access */
        T * data() { return m_buffer.get(); }

        /** Returns the size of the image content */
        size_t size() const
        {
            return m_stride*m_height;
        }

        /** Image copy constructor */
        Image(Image const& other)
        {
            resize(other.m_height, other.m_width, other.m_stride);

            memcpy(m_buffer.get(), other.m_buffer.get(), size());
        }

        /** Image move constructor */
        Image(Image && other)
        {
            m_height = other.m_height;
            m_width  = other.m_width;
            m_stride = other.m_stride;
            m_buffer = std::move(other.m_buffer);

            other.m_height = 0;
            other.m_width  = 0;
            other.m_stride = 0;
            other.m_buffer.reset();
        }

	private:
		size_t m_height;  ///< Image height
		size_t m_width;   ///< Image width
		size_t m_stride;  ///< Image stride

        std::unique_ptr<T[]> m_buffer;  ///< Image buffer
	};
}

// End definition
#endif // VOX_IMAGE_H
