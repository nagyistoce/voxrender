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

// http://www.alexander-noe.com/video/documentation/avi.pdf

// Include Header
#include "AviVid.h"

// Include Dependencies
#include "VoxLib/Core/Common.h"
#include "VoxLib/Core/Functors.h"
#include "VoxLib/Core/Geometry/Vector.h"
#include "VoxLib/Core/Logging.h"

// API namespace
namespace vox
{

namespace {
namespace filescope {

    FourCC RIFF = FourCC('R', 'I', 'F', 'F');
    FourCC LIST = FourCC('L', 'I', 'S', 'T');
    
    FourCC FOURCC_AVI  = FourCC('A', 'V', 'I', ' ');
    FourCC FOURCC_AVIX = FourCC('A', 'V', 'I', 'X');
    FourCC FOURCC_VIDS = FourCC('v', 'i', 'd', 's');
    FourCC FOURCC_AUDS = FourCC('a', 'u', 'd', 's');
    FourCC FOURCC_TXTS = FourCC('t', 'x', 't', 's');

    // AVI chunk atom
    struct Chunk
    {
        UInt32 fourCC;
        UInt32 size;    ///< Size of chunk data
        // UInt8 * Data //
    };

    // AVI list atom
    struct List
    {
        UInt32 list;
        UInt32 size;
        FourCC fourCC;
        // UInt8 * Data //
    };

    // AVI file header
    struct Header
    {
        UInt32 microSecPerFrame;    ///< Frame display rate (generally unreliable)
        UInt32 maxBytesPerSec;      ///< Highest data rate demanded for playback
        UInt32 paddingGranularity;  ///< The file is padded to a multiple of this size

        UInt32 flags;               ///< Generic flags
        UInt32 totalFrames;         ///< Frames in ONLY the RIFF-AVI list (generally unreliable)
        UInt32 initialFrames;       ///< WTF 
        UInt32 streams;             ///< Number of data streams in the file
        UInt32 suggestedBufferSize; ///< Suggested buffer size during read

        UInt32 width;   ///< Width of the video stream
        UInt32 height;  ///< Height of the video stream

        UInt32 reserved[4];
    };

    // Stream header list element
    struct StreamHeader
    {
        FourCC fccType;             ///< Stream type: vids, auds or text
        FourCC fccHandler;          ///< FourCC of the codec to be used
        UInt32 flags;       
        UInt16 priority;
        UInt16 language;
        UInt32 initialFrames;       ///< Number of the first block of the stream that is present in the file
        UInt32 scale;
        UInt32 rate;                ///< Rate / scale = samples / second (should be mutually prime for bad players)
        UInt32 start;               ///< Start time of the stream ('silent' frames before the stream starts)
        UInt32 length;              ///< Size of stream
        UInt32 suggestedBufferSize; ///< Buffer size necessary to store 1 block (should be non-zero for bad players)
        UInt32 quality;             ///< (Unimportant) stream quality
        UInt32 sampleSize;          ///< Minimum number of bytes in one stream atom
        Int32  frame[4];
    };

} // namespace filescope
} // namespace anonymous

} // namespace vox