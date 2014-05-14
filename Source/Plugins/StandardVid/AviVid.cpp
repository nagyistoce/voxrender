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

} // namespace filescope
} // namespace anonymous

/* 5 seconds stream duration */
#define STREAM_DURATION   5.0
#define STREAM_FRAME_RATE 25 /* 25 images/s */
#define STREAM_NB_FRAMES  ((int)(STREAM_DURATION * STREAM_FRAME_RATE))
#define STREAM_PIX_FMT PIX_FMT_YUV420P /* default pix_fmt */

// ----------------------------------------------------------------------------
//  Constructor
// ----------------------------------------------------------------------------
AviWriter::AviWriter(std::shared_ptr<void> handle) : 
    m_handle(handle), m_frameCount(0)
{ 
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
void AviWriter::begin(ResourceOStream & ostr, OptionSet const& options)
{
    auto filename = ostr.identifier().path.c_str();

    // Initialize the output media context
    auto format = av_guess_format(nullptr, filename, nullptr);
    if (!format)
    {
        throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY, 
            "Call to av_new_stream failed", Error_Unknown);
    }
    m_oc = avformat_alloc_context();
    if (!m_oc) 
    {
        throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY, 
            "Call to avformat_alloc_context failed", Error_NoMemory);
    }
    m_oc->oformat = format;

    // Add the audio and video streams as necessary
    if (format->video_codec != CODEC_ID_NONE) addVideoStream();
    if (format->audio_codec != CODEC_ID_NONE) addAudioStream();
    if (format->subtitle_codec != CODEC_ID_NONE) addSubStream();
    // :TODO: Subtitles/multiple audios stuff/etc

    // Set parameters

    // :DDEBUG: dump of format information
    //av_dump_format(m_oc, 0, filename, 1);

    // Open the codecs    
    if (format->video_codec != CODEC_ID_NONE) openVideo();
    if (format->audio_codec != CODEC_ID_NONE) ;
    if (format->subtitle_codec != CODEC_ID_NONE) ;

    // Open the output file, if needed
    if (!(format->flags & AVFMT_NOFILE)) 
    {
        if (avio_open(&m_oc->pb, filename, AVIO_FLAG_WRITE) < 0) 
        {
            throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY, 
                "Call to avio_open failed", Error_Unknown);
        }
    }

    // Write the stream header, if any
    avformat_write_header(m_oc, nullptr);
}

// ----------------------------------------------------------------------------
//  Adds an audio stream to the AV file
// ----------------------------------------------------------------------------
void AviWriter::allocPicture()
{
    auto c = m_videoSt->codec;

    if (!(m_picture = av_frame_alloc()))
    {
        throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY, 
            "Call to avcodec_alloc_frame failed", Error_Unknown);
    }

    if (avpicture_alloc((AVPicture*)m_picture, c->pix_fmt, c->width, c->height))
    {
        throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY, 
            "Call to avpicture_alloc failed", Error_Unknown);
    }
}

// ----------------------------------------------------------------------------
//  Adds an audio stream to the AV file
// ----------------------------------------------------------------------------
void AviWriter::addAudioStream()
{
}

// ----------------------------------------------------------------------------
//  Adds a subtitle stream to the AV file
// ----------------------------------------------------------------------------
void AviWriter::addSubStream()
{
}

// ----------------------------------------------------------------------------
//  Adds a video stream to the AV file
// ----------------------------------------------------------------------------
void AviWriter::addVideoStream()
{
     AVCodecContext * c;
 
     // Create the video output stream
     if (!(m_videoSt = avformat_new_stream(m_oc, 0))) 
     {
        throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY, 
            "Call to av_new_stream failed", Error_Unknown);
     }
 
     // Specify the video codec parameters
     c = m_videoSt->codec;
     c->codec_id = m_oc->oformat->video_codec; ///< libAV codec id
     c->codec_type = AVMEDIA_TYPE_VIDEO;       ///< The type of stream
     c->bit_rate = 400000;                     ///< Target bitrate for the stream
     c->width    = 512;                        ///< Image width in the stream
     c->height   = 512;                        ///< Image height in the stream
     c->time_base.den = STREAM_FRAME_RATE;
     c->time_base.num = 1;                     ///< Time step size per stamp in seconds (num / den)
     c->gop_size = 12;                         ///< Emit one intra frame at most every X frames
     c->pix_fmt = STREAM_PIX_FMT;              ///< Pixel format of the underlying video data
     if (c->codec_id == CODEC_ID_MPEG1VIDEO)
     {
         // Needed to avoid using macroblocks in which some coeffs overflow.
         // This does not happen with normal video, it just happens here as
         // the motion of the chroma plane does not match the luma plane. 
         c->mb_decision=2;
     }

     // some formats want stream headers to be separate
     if(m_oc->oformat->flags & AVFMT_GLOBALHEADER)
         c->flags |= CODEC_FLAG_GLOBAL_HEADER;

     // Allocate memory for the intermediary frames
     allocPicture();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
void AviWriter::openVideo()
{
    auto c = m_videoSt->codec;
 
    // Find the video encoder
    auto codec = avcodec_find_encoder(c->codec_id);
    if (!codec) 
    {
        throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY,
            "Call to avcodec_find_encoder failed", Error_Unknown);
    }
 
    // Open the codec
    if (avcodec_open2(c, codec, nullptr) < 0) 
    {
        throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY,
            "Call to avcodec_open2 failed", Error_Unknown);
    }
 
    // Allocate the encoded raw picture 
    allocPicture();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
void AviWriter::closeVideo()
{
    avcodec_close(m_videoSt->codec);
    av_free(m_picture->data[0]);
    av_free(m_picture);
}

// ----------------------------------------------------------------------------
//  Adds a frame of video to the output stream
// ----------------------------------------------------------------------------
void AviWriter::addFrame(ResourceOStream & ostr, Bitmap const& bitmap)
{
    AVCodecContext * c = m_videoSt->codec;

    if (m_frameCount >= STREAM_NB_FRAMES) return;
 
    auto dptr = (UInt8*)bitmap.data();
    auto ptr = (UInt8*)bitmap.data();
    for (int j = 0; j < bitmap.height(); j++)
    {
        for (int i = 0; i < bitmap.width(); i++)
        {
            *dptr++ = *ptr++;
            *dptr++ = *ptr++;
            *dptr++ = *ptr++;
            ptr++;
        }
        ptr+=4;
    }

    // Determine the src format so we can perform a conversion
    AVPixelFormat srcFormat;
    switch (bitmap.type())
    {
    case Bitmap::Format_RGBA:
    case Bitmap::Format_RGB:
        if (bitmap.depth() == 8)
        {
            srcFormat = PIX_FMT_RGB24;
            break;
        }
    default:
        throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY,
            "Invalid image format", Error_Unknown);
    }

    // Convert the video frame into the output format
    SwsContext * convertCtx = sws_getCachedContext(nullptr, 
        bitmap.width(), bitmap.height(),
        srcFormat,
        c->width, c->height,
        c->pix_fmt,
        SWS_BICUBIC, nullptr, nullptr, nullptr);

    if (!convertCtx) 
    {
        throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY, 
            "Call to sws_getContext failed", Error_Unknown);
    }

    AVFrame inframe;
    avpicture_fill((AVPicture*)&inframe, (UInt8*)bitmap.data(), srcFormat, bitmap.width(), bitmap.height());

    sws_scale(convertCtx, inframe.data, inframe.linesize,
        0, c->height, m_picture->data, m_picture->linesize);
    
    // Push the next frame into the encoder
    writeFrame(m_picture);

    m_frameCount++;
}

// ----------------------------------------------------------------------------
//  Writes video to the encoder and the output through the interleaver
// ----------------------------------------------------------------------------
bool AviWriter::writeFrame(AVFrame * frame)
{
    AVPacket pkt;
    av_init_packet(&pkt);
    pkt.data = nullptr;
    pkt.size = 0;
    int got;
    if (avcodec_encode_video2(m_videoSt->codec, &pkt, frame, &got))
    {
        throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY, 
            "Call to avcodec_encode_video2 failed", Error_Unknown);
    } 

    // Write the next packet to the stream if got
    if (got && av_interleaved_write_frame(m_oc, &pkt))
    {
        throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY, 
            "Call to av_interleaved_write_frame failed", Error_Unknown);
    }
    av_free_packet(&pkt); 

    return got;
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
void AviWriter::end(ResourceOStream & ostr)
{
    // Ensure the encoder pipeline is empty
    while (writeFrame(nullptr)) ;

    // Write the trailer
    av_write_trailer(m_oc);

    // Close the output file
    if (!(m_oc->oformat->flags & AVFMT_NOFILE)) 
    {
        avio_close(m_oc->pb);
    }

    //
    closeVideo();

    // Free the contexts
    for(int i = 0; i < m_oc->nb_streams; i++) 
    {
        av_freep(&m_oc->streams[i]->codec);
        av_freep(&m_oc->streams[i]);
    }
    av_free(m_oc);

    // Close the output stream
    ostr.close();
}

} // namespace vox