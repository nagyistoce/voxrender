/* ===========================================================================

	Project: VoxRender - Frame Buffer

	Description: Render frame buffer and callback info structure

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
#ifndef VOX_FRAME_BUFFER_H
#define VOX_FRAME_BUFFER_H

// Include Dependencies
#include "VoxLib/Core/Common.h"
#include "VoxLib/Core/Geometry/Image.h"
#include "VoxLib/Core/Geometry/Color.h"

// API namespace
namespace vox
{
	/** Rendering frame buffer class */
    class VOX_EXPORT FrameBuffer : public Image<ColorRgbaLdr>
	{
	public:
        /** Constructs and image of the specified dimensions */
		FrameBuffer(size_t width, size_t height) :
            m_locked(false), Image(width, height, sizeof(ColorRgbaLdr)*width)
        { 
        }

        /** Image constructor controlled by the renderer */
		FrameBuffer(size_t width, size_t height, size_t stride) :
            m_locked(false), Image(width, height, stride)
        { 
        }

        /** Waits for access to the image buffer */
        void wait() 
        { 
            if (m_locked)
            {
                boost::mutex::scoped_lock lock(m_mutex);
                while (m_locked) m_cond.wait(lock);
            }
        }

	private:
        friend class FrameBufferLock;

        boost::condition_variable m_cond;   ///< Lock synchronization condition
        boost::mutex			  m_mutex;  ///< Lock synchronization mutex
        bool					  m_locked; ///< Lock flag for image callback

        /** Locks the image buffer for the user callback */
        void lock() { m_locked = true; }

        /** Unlocks the image buffer for the renderer */
        void unlock() 
        {
            boost::mutex::scoped_lock lock(m_mutex);
            m_locked = false;
            m_cond.notify_all();
        }
	};

    /** Render callback structure */
    class VOX_EXPORT FrameBufferLock
    {
    public:
        /** Initiates a lock around the input framebuffer */
        FrameBufferLock(std::shared_ptr<FrameBuffer> buffer) :
            framebuffer(buffer)
        {
            framebuffer->lock();
        }

        /** Releases the lock on the held framebuffer */
        ~FrameBufferLock() { framebuffer->unlock(); }

        std::shared_ptr<FrameBuffer> framebuffer;   ///< Frame buffer

    private:
        FrameBufferLock(FrameBuffer & other);
    };
}

// End definition
#endif // VOX_FRAME_BUFFER_H
