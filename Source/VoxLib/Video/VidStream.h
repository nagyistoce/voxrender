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
#include "VoxLib/Video/VidCodec.h"
#include "VoxLib/Bitmap/Bitmap.h"

// API namespace
namespace vox
{
	/** Audio-Video output stream */
	class VOX_EXPORT VidOStream
	{
    public:
        /** Constructor */
        VidOStream() { }

        /** Constructor */
        VidOStream(ResourceOStream & output, OptionSet options = OptionSet(), String const& format = "")
        {
            open(output, options, format);
        }
        
        /** Constructor */
        VidOStream(ResourceId const& output, OptionSet options = OptionSet(), String const& format = "")
        {
            open(output, options, format);
        }

        /** Wraps an existing IO stream with a video encoder */
        void open(ResourceOStream & output, OptionSet options = OptionSet(), String const& format = "");

        /** 
         * Opens the specified resource and wraps it with a video encoder 
         *
         * This function initializes a video output stream and establishes a filter chain for
         * decoding the data as it is read using the push function.
         */
        void open(ResourceId const& uri, OptionSet options = OptionSet(), String const& format = "");

        /** Marks the end of the video output */
        void close();

        /** Returns true if the video is opened */
        bool isOpen();

        /** Pushes a singe frame into the video stream */
        void push(Bitmap const& frame, int streamId = -1);

        /** Stream style method for calling push */
        void operator<<(Bitmap const& frame) { push(frame); }

        /** Registers a video encoder to be associated with the specified format */
        static void registerEncoder(String const& format, std::shared_ptr<VideoEncoder> encoder);

        /** Removes an encoder from the internal mapping of encoders to formats */
        static void removeEncoder(std::shared_ptr<VideoEncoder> encoder);
        
        /** Removes the encoder for a given format from the internal mapping */
        static void removeEncoder(String const& format, std::shared_ptr<VideoEncoder> encoder = nullptr);

        /** Returns a list of the registered encoder formats */
        static std::list<String> encoders();

    private:
        /** Acquires a writer suitable for the stream's format */
        void getWriter(String const& format);

    private:
        std::shared_ptr<VideoWriter> m_device;

        std::shared_ptr<ResourceOStream> m_ostr;
	};

    /** Audio-Video input stream */
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
        
        /** Returns true if the video is opened */
        bool isOpen();

        /** Performs a high level seek of the video data */
        void seek();

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
        /** Acquires a reader suitable for the stream's format */
        void getReader(String const& format);

    private:
        std::shared_ptr<VideoReader> m_device;
        
        std::unique_ptr<ResourceIStream> m_istr;
    };
}

// End definition
#endif // VOX_VID_STREAM_H