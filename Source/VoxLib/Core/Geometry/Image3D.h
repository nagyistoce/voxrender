/* ===========================================================================

	Project: VoxRender - Image3D class

	Description: Defines the 3D image class for rendering operations

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
#ifndef VOX_IMAGE3D_H
#define VOX_IMAGE3D_H

// Include Dependencies
#include "VoxLib/Core/CudaCommon.h"
#include "VoxLib/Core/Types.h"

// API namespace
namespace vox
{
	/** Rendering frame buffer class */
    template<typename T> class Image3D
	{
	public:
		/** Initializes an empty image structure */
		Image3D() : m_width(0), m_height(0), m_depth(0), m_buffer(nullptr) { }

        /** Constructs and image of the specified dimensions */
		Image3D(size_t width, size_t height, size_t depth) :
            m_width(width), m_height(height), m_depth(depth),
            m_buffer(reinterpret_cast<T*>(new UInt8[width*height*depth*sizeof(T)])) 
        { 
        }

        /** Resizes the image to the specified dimensions */
        void resize(size_t width, size_t height, size_t depth)
        {
            m_width = width; m_height = height; m_depth = depth;

            m_buffer = std::unique_ptr<T[]>(reinterpret_cast<T*>(new UInt8[width*depth*height*sizeof(T)]));
        }

        /** Zeroes the image data */
        void clear() { memset(m_buffer.get(), 0, size()*sizeof(T)); }

        /** Image width accessor */
        size_t width() const { return m_width; }

        /** Image height accessor */
        size_t height() const { return m_height; }

        /** Image depth accessor */
        size_t depth() const { return m_depth; }

        /** Image buffer accessor for const data access */
        T const* data() const { return m_buffer.get(); }

        /** Image buffer accessor for mutable data access */
        T * data() { return m_buffer.get(); }

        /** Voxel accessor for mutable data access */
        T & at(size_t x, size_t y, size_t z)
        {
            auto p = m_width * m_height;
            return m_buffer.get()[x + y*m_width + z*p];
        }

        /** Voxel accessor for const data access */
        T const& at(size_t x, size_t y, size_t z) const
        {
            auto p = m_width * m_height;
            return m_buffer.get()[x + y*m_width + z*p];
        }

        /** Returns the size of the image content */
        size_t size() const
        {
            return m_width*m_height*m_depth;
        }

        /** Image copy constructor */
        Image3D& operator=(Image3D const& other)
        {
            resize(other.width(), other.height(), other.depth());

            memcpy(m_buffer.get(), other.data(), size()*sizeof(T));

            return *this;
        }

        /** Image assignment operator */
        Image3D(Image3D const& other)
        {
            resize(other.m_width, other.m_height, other.m_depth);

            memcpy(m_buffer.get(), other.m_buffer.get(), size()*sizeof(T));
        }

        /** Image move constructor */
        Image3D(Image3D && other)
        {
            m_height = other.m_height;
            m_width  = other.m_width;
            m_depth  = other.m_depth;
            m_buffer = std::move(other.m_buffer);

            other.m_height = 0;
            other.m_width  = 0;
            other.m_depth  = 0;
            other.m_buffer.reset();
        }

	private:
		size_t m_width;   ///< Image width
		size_t m_height;  ///< Image height
        size_t m_depth;   ///< Image depth

        std::unique_ptr<T[]> m_buffer;  ///< Image buffer
	};
}

// End definition
#endif // VOX_IMAGE3D_H
