/* ===========================================================================
                                                                           
   Project: VoxLib                                    

   Description: Handles streaming of video data                      

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
#ifndef VOX_VID_CODEC_H
#define VOX_VID_CODEC_H

// Include Dependencies
#include "VoxLib/Bitmap/Bitmap.h"
#include "VoxLib/Core/CudaCommon.h"
#include "VoxLib/Core/Geometry/Vector.h"
#include "VoxLib/Error/Error.h"
#include "VoxLib/IO/ResourceId.h"
#include "VoxLib/IO/Resource.h"

// API Namespace
namespace vox
{

typedef UInt32 FourCC; ///< FourCC compression code

/** Handles the writing of video content to a stream */
class VOX_EXPORT VideoWriter
{
public:
    virtual ~VideoWriter() { }

    virtual void begin(ResourceOStream & ostr, OptionSet const& options) { }

    virtual void addFrame(Bitmap const& frame) { }
    
    virtual void end(ResourceOStream & ostr) { }
};

/** Handles the reading of video content from a stream */
class VOX_EXPORT VideoReader
{
public:
    virtual ~VideoReader() { }

    virtual void begin(ResourceIStream & ostr, OptionSet const& options) { }
    
    virtual void getFrame() { }

    virtual void end(ResourceIStream & ostr) { }
};

/** Supplies video writers for OStreams */
class VOX_EXPORT VideoEncoder
{
public:
    virtual ~VideoEncoder() { }

    virtual std::shared_ptr<VideoWriter> writer() = 0;
};

/** Supplies video readers for IStreams */
class VOX_EXPORT VideoDecoder
{
public:
    virtual ~VideoDecoder() { }

    virtual std::shared_ptr<VideoReader> reader() = 0;
};

/** Convenience class for both encode and decode support */
class VideoCodec : public VideoEncoder, public VideoDecoder { };

}

// End Definition
#endif // VOX_VID_CODEC_H