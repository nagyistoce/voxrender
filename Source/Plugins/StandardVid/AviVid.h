/* ===========================================================================

    Project: Standard Video Import/Export
    
	Description: Provides an import/export module for video formats

    Copyright (C) 2014 Lucas Sherman

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
#ifndef VOX_SVID_AVI_VID_H
#define VOX_SVID_AVI_VID_H

// Include Dependencies
#include "StandardVid/Common.h"
#include "VoxLib/Core/Format.h"
#include "VoxLib/Video/VidCodec.h"
#include "VoxLib/IO/Resource.h"

// API namespace
namespace vox 
{
    
/** Writes an AVI format video file */
class VOX_SVID_EXPORT AviWriter : public VideoWriter
{
public:
    AviWriter(std::shared_ptr<void> handle);

    virtual void begin(ResourceOStream & ostr, OptionSet const& options);

    virtual void end(ResourceOStream & ostr);

    virtual void addFrame(Bitmap const& bitmap);

private:
    void addIndexEntry();

private:
    std::shared_ptr<void> m_handle;

    std::streamsize m_indexPos; ///< Position of the write head for the video index header
};

/** Reads an AVI format video file */
class VOX_SVID_EXPORT AviReader : public VideoReader
{
public:
    AviReader(std::shared_ptr<void> handle) : m_handle(handle) { }
    
    virtual void begin(ResourceOStream & ostr, OptionSet const& options) { }

    virtual void end(ResourceOStream & ostr) { }

    virtual void getFrame() { }

private:
    std::shared_ptr<void> m_handle;
};

/**
 * Standard video file import / export module
 *
 * This module is compatible with the abstract video codec interface
 */
class VOX_SVID_EXPORT AviVid : public VideoCodec
{
public:
    AviVid(std::shared_ptr<void> handle) : m_handle(handle) { }
    
    std::shared_ptr<VideoWriter> writer() { return std::make_shared<AviWriter>(m_handle); }
    
    std::shared_ptr<VideoReader> reader() { return std::make_shared<AviReader>(m_handle); }

private:
    std::shared_ptr<void> m_handle; ///< Plugin handle to track this DLL's usage
};

}

// End definition
#endif // VOX_SVID_AVI_VID_H