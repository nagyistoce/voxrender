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
    auto path = ostr.identifier().path;
    if (path.front() == '/') path = path.substr(1);
    auto filename = path.c_str();

    // Initialize the output media context
    auto format = av_guess_format(nullptr, filename, nullptr);
    if (!format)
    {
        throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY, 
            "Call to av_new_stream failed", Error_Unknown);
    }
    m_oc.reset(avformat_alloc_context(), &av_free);
    if (!m_oc) 
    {
        throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY, 
            "Call to avformat_alloc_context failed", Error_NoMemory);
    }
    m_oc->oformat = format;

    // Initialize the requested number of AV streams
    if (format->video_codec != CODEC_ID_NONE)
    {
        m_videoStreams.resize(options.lookup<unsigned int>("videoStreams", 1));
        BOOST_FOREACH (auto & stream, m_videoStreams) stream = addVideoStream(options);
    }
    if (format->audio_codec != CODEC_ID_NONE) addAudioStream(options);
    if (format->subtitle_codec != CODEC_ID_NONE) addSubStream(options);

    // Open the output file, if needed
    if (!(format->flags & AVFMT_NOFILE)) 
    {
        if (avio_open(&m_oc->pb, filename, AVIO_FLAG_WRITE) < 0) 
        {
            throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY, 
                "Call to avio_open failed", Error_Unknown);
        }
    }

    allocPicture();

    // Write the stream header, if any
    avformat_write_header(m_oc.get(), nullptr);
}

// ----------------------------------------------------------------------------
//  Adds an audio stream to the AV file
// ----------------------------------------------------------------------------
void AviWriter::allocPicture()
{
    auto c = m_videoStreams[0]->codec;

    m_picture.reset(av_frame_alloc(), &av_free);
    if (!m_picture)
    {
        throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY, 
            "Call to avcodec_alloc_frame failed", Error_Unknown);
    }

    if (avpicture_alloc((AVPicture*)m_picture.get(), c->pix_fmt, c->width, c->height))
    {
        throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY, 
            "Call to avpicture_alloc failed", Error_Unknown);
    }
}

// ----------------------------------------------------------------------------
//  Adds an audio stream to the AV file
// ----------------------------------------------------------------------------
void AviWriter::addAudioStream(OptionSet const& options)
{
}

// ----------------------------------------------------------------------------
//  Adds a subtitle stream to the AV file
// ----------------------------------------------------------------------------
void AviWriter::addSubStream(OptionSet const& options)
{
    AVStream * stream;
    
     // Create the subtitle output stream
     if (!(stream = avformat_new_stream(m_oc.get(), 0))) 
     {
        throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY, 
            "Call to av_new_stream failed", Error_Unknown);
     }
     
     // Specify the subtitle codec parameters
     auto c = stream->codec;
     c->codec_id = m_oc->oformat->subtitle_codec;   ///< LibAV codec id
     c->codec_type = AVMEDIA_TYPE_SUBTITLE;         ///< The type of the stream

     // some formats want stream headers to be separate
     if(m_oc->oformat->flags & AVFMT_GLOBALHEADER)
         c->flags |= CODEC_FLAG_GLOBAL_HEADER;
}

// ----------------------------------------------------------------------------
//  Adds a video stream to the AV file
// ----------------------------------------------------------------------------
AVStream* AviWriter::addVideoStream(OptionSet const& options)
{
    AVStream* stream;
 
    // Create the video output stream
    if (!(stream = avformat_new_stream(m_oc.get(), 0))) 
    {
        throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY, 
            "Call to av_new_stream failed", Error_Unknown);
    }
 
    // Specify the video codec parameters
    auto c = stream->codec;
    c->codec_id = m_oc->oformat->video_codec;              ///< libAV codec id
    c->codec_type = AVMEDIA_TYPE_VIDEO;                    ///< The type of stream
    c->bit_rate = options.lookup<int>("bitrate", 2500000); ///< Target bitrate for the stream (default = 2.5Mbs broadband average)
    c->width    = options.lookup<int>("width", 640);       ///< Image width in the stream
    c->height   = options.lookup<int>("height", 480);      ///< Image height in the stream
    c->time_base.den = options.lookup<int>("framerate", 30);
    c->time_base.num = 1;                                  ///< Time step size per stamp in seconds (num / den)
    c->gop_size = 12;                                      ///< Emit one intra frame at most every X frames
    c->pix_fmt = PIX_FMT_YUV420P;                          ///< Pixel format of the underlying video data

    // some formats want stream headers to be separate
    if(m_oc->oformat->flags & AVFMT_GLOBALHEADER)
        c->flags |= CODEC_FLAG_GLOBAL_HEADER;

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

    return stream;
}

// ----------------------------------------------------------------------------
//  Adds a frame of video to the output stream
// ----------------------------------------------------------------------------
void AviWriter::addFrame(ResourceOStream & ostr, Bitmap const& bitmap, int streamId)
{
    if (streamId > 0 && streamId > m_videoStreams.size()) throw 
        Error(__FILE__, __LINE__, VOX_LOG_CATEGORY, "streamId out of index", Error_Range);

    // Determine the src format so we can perform a conversion
    AVPixelFormat srcFormat;
    if (bitmap.depth() == 8)
    {
        switch (bitmap.type())
        {
        case Bitmap::Format_RGBA:      srcFormat = AV_PIX_FMT_RGBA;    break;
        case Bitmap::Format_RGB:       srcFormat = AV_PIX_FMT_RGB8;    break;
        case Bitmap::Format_Gray:      srcFormat = AV_PIX_FMT_GRAY8;   break;
        case Bitmap::Format_RGBX:      srcFormat = AV_PIX_FMT_RGB0;    break;
        case Bitmap::Format_GrayAlpha: srcFormat = AV_PIX_FMT_GRAY8A;  break;
        default:
            throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY,
                "Invalid image format", Error_Unknown);
        }
    }
    else
    {
        throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY,
            "Invalid image format", Error_Unknown);
    }
    
    // streamId negative => auto detect ID
    unsigned int front = 0; 
    unsigned int back  = 0;
    if (streamId < 0)
    {
        // Make sure the layers and stream counts match on auto
        if (bitmap.layers() != 1 && m_videoStreams.size() != 1) 
        {
            if (m_videoStreams.size() != bitmap.layers()) 
                throw Error(__FILE__, __LINE__, VOX_SVID_LOG_CATEGORY, 
                    "Bitmap layer count and video stream count must match.");
            back = m_videoStreams.size() - 1;
        }
    }
    else front = back = streamId;

    // Process each layer of the next frame
    for (auto sId = front; sId <= back; sId++)
    {
        auto c = m_videoStreams[sId]->codec;

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
        avpicture_fill((AVPicture*)&inframe, (UInt8*)bitmap.data(sId), srcFormat, bitmap.width(), bitmap.height());

        sws_scale(convertCtx, inframe.data, inframe.linesize,
            0, c->height, m_picture->data, m_picture->linesize);
    
        // Push the next frame into the encoder
        writeFrame(m_picture.get(), sId);
    }
}

// ----------------------------------------------------------------------------
//  Writes video to the encoder and the output through the interleaver
// ----------------------------------------------------------------------------
int AviWriter::writeFrame(AVFrame * frame, unsigned int streamId)
{
    AVPacket pkt;
    av_init_packet(&pkt);
    pkt.data = nullptr;
    pkt.size = 0;
    std::shared_ptr<AVPacket> pktPtr(&pkt, &av_free_packet); 

    int got;
    if (avcodec_encode_video2(m_videoStreams[streamId]->codec, &pkt, frame, &got))
    {
        throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY, 
            "Call to avcodec_encode_video2 failed", Error_Unknown);
    } 
    pkt.stream_index = streamId;

    // Write the next packet to the stream if got
    if (got && av_interleaved_write_frame(m_oc.get(), &pkt))
    {
        throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY, 
            "Call to av_interleaved_write_frame failed", Error_Unknown);
    }

    return got;
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
void AviWriter::end(ResourceOStream & ostr)
{
    // Ensure the encoder pipeline is empty
    for (unsigned int i = 0; i < m_videoStreams.size(); i++)
        while (writeFrame(nullptr, i)) ;

    // Write the trailer
    av_write_trailer(m_oc.get());

    // Close the output file
    if (!(m_oc->oformat->flags & AVFMT_NOFILE)) 
    {
        avio_close(m_oc->pb);
    }

    // Free the contexts
    for(unsigned int i = 0; i < m_oc->nb_streams; i++) 
    {
        avcodec_close(m_oc->streams[i]->codec);
        av_freep(&m_oc->streams[i]->codec);
        av_freep(&m_oc->streams[i]);
    }

    // Close the output stream
    ostr.close();
}

} // namespace vox