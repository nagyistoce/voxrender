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
#include "VoxLib/Core/Functors.h"

// API namespace
namespace vox
{
	/** Rendering frame buffer class */
    template<typename T> class Image
	{
	public:
		/** Initializes an empty image structure */
		Image() : m_width(0), m_height(0), m_stride(0), m_buffer(nullptr) { }

        /** Constructs and image of the specified dimensions */
		Image(size_t width, size_t height) :
            m_width(width), m_height(height), 
            m_stride(sizeof(T)*width),
            m_buffer(new T[width*height], arrayDeleter) 
        { 
        }

        /** Constructs and image of the specified dimensions */
		Image(size_t width, size_t height, size_t stride) :
            m_width(width), m_height(height), m_stride(stride),
            m_buffer(reinterpret_cast<T*>(new UInt8[stride*height]), arrayDeleter) 
        { 
        }
          
        /** Constructs and image of the specified dimensions */
		Image(size_t width, size_t height, size_t stride, std::shared_ptr<T> buffer) :
            m_width(width), m_height(height), m_stride(stride), m_buffer(buffer) 
        { 
        }

        /** Clears the image data */
        void clear(T const& val = T())
        {
	        for (size_t j = 0; j < m_height; j++)
	        for (size_t i = 0; i < m_width;  i++)
                at(i, j) = val;
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

            m_buffer.reset(reinterpret_cast<T*>(new UInt8[stride*height]), arrayDeleter);
        }

        /** Image width accessor */
        size_t width() const { return m_width; }

        /** Image height accessor */
        size_t height() const { return m_height; }

        /** Image stride accessor */
        size_t stride() const { return m_stride; }

        /** Image pixel accessor */
        T & at(size_t x, size_t y) { return *((T*)(((UInt8*)m_buffer.get()) + y * m_stride + x * sizeof(T))); }
        
        /** Image pixel accessor */
        T const& at(size_t x, size_t y) const { return *((T*)(((UInt8*)m_buffer.get()) + y * m_stride + x * sizeof(T))); }
        
        /** Image buffer accessor for const data access */
        T const* data() const { return m_buffer.get(); }

        /** Image buffer accessor for mutable data access */
        T * data() { return m_buffer.get(); }

        /** Returns the size of the image content */
        size_t size() const
        {
            return m_stride*m_height;
        }

        /** Returns the internal managed buffer */
        std::shared_ptr<T> buffer() { return m_buffer; }

        /** Image clone functionality */
        Image copy()
        {
            Image result(m_width, m_height, m_stride);

            memcpy(result.data(), m_buffer.get(), size());

            return result;
        }

	private:
        size_t m_height;  ///< Image height
		size_t m_width;   ///< Image width
		size_t m_stride;  ///< Image stride

        std::shared_ptr<T> m_buffer;  ///< Image buffer
	};
}

// End definition
#endif // VOX_IMAGE_H
