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
#ifndef VOX_VID_STREAM_H
#define VOX_VID_STREAM_H

// Include Dependencies
#include "VoxLib/Core/Common.h"
#include "VoxLib/IO/Resource.h"

// API namespace
namespace vox
{
    class VideoEncoder;
    class VideoDecoder;

	/** 
	 * Video output stream
	 */
	class VOX_EXPORT VidOStream
	{
    public:
        /** Wraps an existing IO stream with a video encoder */
        void open(ResourceOStream & output, OptionSet options = OptionSet(), String const& format = "");

        /** 
         * Opens the specified resource and wraps it with a video encoder 
         *
         * This function initializes a video output stream and establishes a filter chain for
         * decoding the data as it is read using the push function.
         */
        void open(ResourceId const& uri, OptionSet options = OptionSet(), String const& format = "");

        /**
         * Terminates the internal filter chain and releases the associated video codec
         */
        void close();

        /** Pushes a singe frame into the video stream */
        void push();

        /** Stream style method for calling push */
        void operator<<(void * frame);

        /** Registers a video encoder to be associated with the specified format */
        static void registerEncoder(String const& format, std::shared_ptr<VideoEncoder> encoder);

        /** Removes an encoder from the internal mapping of encoders to formats */
        static void removeEncoder(std::shared_ptr<VideoEncoder> encoder);
        
        /** Removes the encoder for a given format from the internal mapping */
        static void removeEncoder(String const& format, std::shared_ptr<VideoEncoder> encoder = nullptr);

    private:
        std::shared_ptr<VideoEncoder> m_encoder;

        String m_format;
	};

    /** */
    class VOX_EXPORT VidIStream
    {
    public: 
        /** Wraps an existing IO stream with a video encoder */
        void open(ResourceOStream & output, OptionSet options = OptionSet(), String const& format = "");

        /** 
         * Opens the specified resource and wraps it with a video encoder 
         *
         * This function initializes a video output stream and establishes a filter chain for
         * decoding the data as it is read using the push function.
         */
        void open(ResourceId const& uri, OptionSet options = OptionSet(), String const& format = "");

        /**
         * Terminates the internal filter chain and releases the associated video codec
         */
        void close();

        /** Pulls a singe frame from the video stream*/
        void pull();

        /** Stream style method for calling pull */
        void operator>>(void * frame);

        /** Registers a video encoder to be associated with the specified format */
        static void registerDecoder(String const& format, std::shared_ptr<VideoDecoder> decoder);

        /** Removes an encoder from the internal mapping of encoders to formats */
        static void removeDecoder(std::shared_ptr<VideoDecoder> decoder);
        
        /** Removes the encoder for a given format from the internal mapping */
        static void removeDecoder(String const& format, std::shared_ptr<VideoDecoder> decoder = nullptr);

        /** The width of a single frame of video */
        unsigned int frameWidth();

        /** The height of a single frame of video */ 
        unsigned int frameHeight();

        /** The framerate the stream */
        float frameRate();

        /** The size in bytes of a single frame */
        unsigned frameSize();

    private:
        std::shared_ptr<VideoDecoder> m_decoder;

        String m_format;
    };

	/** 
	 * Video File Importer
     *
     * @sa
     *  ::VideoExporter
	 */              
    class VideoEncoder 
    {
    public:
    };

    /**
	 * Video File Exporter
     *
     * @sa
     *  ::VideoImporter
     */
    class VideoDecoder 
    { 
    public:
    };

    /** Convenience class for both encode and decode support */
    class VideoCodec : public VideoEncoder, public VideoDecoder { };
}

// End definition
#endif // VOX_VID_STREAM_H