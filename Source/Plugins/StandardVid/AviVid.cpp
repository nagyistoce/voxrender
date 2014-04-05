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

    FourCC RIFF = 'FFIR';
    FourCC LIST = 'TSIL';
    
    FourCC AVI  = ' IVA';
    FourCC AVIX = 'XIVA';
    FourCC HDRL = 'lrdh';
    FourCC STRL = 'lrts';
    FourCC STRH = 'hrts';
    FourCC STRF = 'frts';
    FourCC STRN = 'nrts';
    FourCC VIDS = 'sdiv';
    FourCC AUDS = 'sdua';
    FourCC TXTS = 'stxt';
    FourCC INDX = 'xdni';
    FourCC AVIH = 'hiva';
    FourCC MOVI = 'ivom';

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
    struct AviHeaderChunk
    {
        UInt32 fourCC;
        UInt32 size;
        UInt32 microSecPerFrame;    ///< Frame display rate (generally unreliable)
        UInt32 maxBytesPerSec;      ///< Highest data rate demanded for playback
        UInt32 paddingGranularity;  ///< The file is padded to a multiple of this size

        UInt32 flags;               ///< Generic flags
        UInt32 totalFrames;         ///< Frames in ONLY the RIFF-AVI list (generally unreliable)
        UInt32 initialFrames;       ///< Delay before audio stream in the file 
        UInt32 streams;             ///< Number of data streams in the file
        UInt32 suggestedBufferSize; ///< Suggested buffer size during read

        UInt32 width;   ///< Width of the video stream
        UInt32 height;  ///< Height of the video stream

        UInt32 reserved[4];

        AviHeaderChunk() 
        { 
            memset(this, 0, sizeof(AviHeaderChunk)); 
            fourCC = AVIH;
            size   = sizeof(AviHeaderChunk) - 2 * sizeof(UInt32);
        }
    };

    // Super index chunk for header list
    struct SuperIndexChunk
    {
        FourCC fourCC;
        UInt32 size;
        Int16  longsPerEntry;
        Int8   indexSubType;
        Int8   indexType;
        UInt32 entriesInUse;
        UInt32 chunkId;
        UInt32 reserved[3];
    };

    // Entry for super index chunk
    struct SuperIndexEntry
    {
        Int64 offset;
        UInt32 size;
        UInt32 duration;
    };

    // AVI indexing chunk
    struct StandardIndexChunk
    {
        UInt32 fourCC;
        UInt32 size;
        Int16  longsPerEntry;
        Int8   indexSubType;
        Int8   indexType;
        UInt32 entriesInUse;
        UInt32 chunkId;
        Int64  baseOffset;
        UInt32 reserved[3];

        StandardIndexChunk() 
        { 
            memset(this, 0, sizeof(StandardIndexChunk)); 
            fourCC = INDX;
        }
    };

    // AVI index entry
    struct IndexEntry
    {
        UInt32 offset;
        UInt32 size;
    };

    // Stream header list element
    struct StreamHeaderChunk
    {
        UInt32 fourCC;
        UInt32 size;
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

        StreamHeaderChunk() 
        { 
            memset(this, sizeof(StreamHeaderChunk), 0); 
            fourCC = STRH;
            size   = sizeof(StreamHeaderChunk) - 2 * sizeof(UInt32);
        }
    };
           
    // Bitmap info header for video list
    struct BitmapInfoHeaderChunk
    {
        UInt32 fourCC;
        UInt32 size;
        UInt32 headerSize;
        UInt32 imageWidth;
        UInt32 imageHeight;
        UInt16 nColorPlanes;
        UInt16 bitsPerPixel;
        UInt32 compressionMethod;
        UInt32 imageSize;
        Int32  horizontalResolution;
        Int32  verticalResolution;
        UInt32 nColors;
        UInt32 nImportantColors;

        BitmapInfoHeaderChunk() 
        { 
            memset(this, 0, sizeof(BitmapInfoHeaderChunk)); 
            fourCC = STRF;
            size   = sizeof(BitmapInfoHeaderChunk) - 2 * sizeof(UInt32);
        }
    };

} // namespace filescope
} // namespace anonymous

// ----------------------------------------------------------------------------
//  Constructor
// ----------------------------------------------------------------------------
AviWriter::AviWriter(std::shared_ptr<void> handle) : m_handle(handle) 
{ 
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
void AviWriter::begin(ResourceOStream & ostr, OptionSet const& options)
{
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
void AviWriter::addFrame(ResourceOStream & ostr, Bitmap const& bitmap)
{
    if (!m_headerWritten) writeHeader(ostr, bitmap.width(), bitmap.height());

    if (bitmap.width() != m_imageSize[0] || bitmap.height() != m_imageSize[1])
    {
        throw Error(__FILE__, __LINE__, VOX_SVID_LOG_CATEGORY,
            "Attempt to write inconsistent image formats to video stream", 
            Error_BadFormat);
    }
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
void AviWriter::end(ResourceOStream & ostr)
{
    // Finalize the chunk/list sizes and indexing

    // Close the output stream
    ostr.close();
}

void AviWriter::writeHeader(ResourceOStream & ostr, unsigned int w, unsigned int h)
{
    m_headerWritten = true;
    m_imageSize[0]  = w;
    m_imageSize[1]  = h;

    // Write the RIFF AVI header list
    filescope::List riff;
    riff.list   = filescope::RIFF;
    riff.size   = 0; // Placeholder (written back on close)
    riff.fourCC = filescope::AVI;
    ostr.write((char*)&riff, sizeof(riff));

    // Write the AVI header list
    filescope::List hdrl;
    hdrl.list   = filescope::LIST;
    hdrl.size   = 0; // Placeholder
    hdrl.fourCC = filescope::HDRL;
    ostr.write((char*)&hdrl, sizeof(hdrl));

    // Write the AVI header
    filescope::AviHeaderChunk avih;
    avih.streams = 1;
    avih.width   = 0; // :TODO:
    avih.height  = 0;

    // Write the stream video header list
    if (true)
    {
        auto size = sizeof(filescope::StreamHeaderChunk) +
                    sizeof(filescope::BitmapInfoHeaderChunk);
        filescope::List strlVideo;
        strlVideo.list   = filescope::LIST;
        strlVideo.size   = size;
        strlVideo.fourCC = filescope::STRL;
        ostr.write((char*)&strlVideo, sizeof(strlVideo));

        filescope::StreamHeaderChunk strh;
        strh.fccType    = filescope::VIDS;
        strh.fccHandler = 0;
        ostr.write((char*)&strh, sizeof(strh));

        filescope::BitmapInfoHeaderChunk strf;
        ostr.write((char*)&strf, sizeof(strf));
    }
    
    // Allocate a block of memory for indexing data
    // :TODO:
    m_indexPos = ostr.tellp();

    // Begin the video data list
    filescope::List movi;
    movi.fourCC = filescope::MOVI;
    movi.list   = filescope::LIST;
    movi.size   = 0;
}

} // namespace vox