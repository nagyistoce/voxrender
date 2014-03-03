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

// Include Header
#include "VidStream.h"

// Include Dependencies
#include "VoxLib/Core/Functors.h"

namespace vox {

namespace {
namespace filescope {
   
    static std::map<String,std::shared_ptr<VideoEncoder>> encoders; // Resource modules
    static std::map<String,std::shared_ptr<VideoDecoder>> decoders; // Resource modules

    static boost::shared_mutex encodeMutex; // Module access mutex for read-write locks
    static boost::shared_mutex decodeMutex; // Module access mutex for read-write locks

} // namespace filescope
} // namespace anonymous

// --------------------------------------------------------------------
//  Opens a resource stream and initializes a video writer
// --------------------------------------------------------------------
void VidOStream::open(ResourceId const& uri, OptionSet options, String const& format)
{
    auto type = format.empty() ? uri.extractFileExtension() : format;
    
    m_ostr = std::shared_ptr<ResourceOStream>(new ResourceOStream(uri));

    getWriter(type);

    m_device->begin(*m_ostr, options);
}

// --------------------------------------------------------------------
//  Opens a resource stream and initializes a video writer
// --------------------------------------------------------------------
void VidOStream::open(ResourceOStream & ostr, OptionSet options, String const& format)
{
    auto type = format.empty() ? ostr.identifier().extractFileExtension() : format;
    
    m_ostr = std::shared_ptr<ResourceOStream>(&ostr, nullDeleter);

    getWriter(type);

    m_device->begin(*m_ostr, options);
}

// --------------------------------------------------------------------
//  Closes the underlying video write device
// --------------------------------------------------------------------
void VidIStream::close()
{
    m_device->end(*m_istr);
    m_device.reset();
    m_istr.reset();
}

// --------------------------------------------------------------------
//  Closes the underlying video write device
// --------------------------------------------------------------------
void VidOStream::close()
{
    m_device->end(*m_ostr);
    m_device.reset();
    m_ostr.reset();
}

// --------------------------------------------------------------------
//  Closes the underlying video read device
// --------------------------------------------------------------------
bool VidIStream::isOpen()
{
    return m_istr;
}

// --------------------------------------------------------------------
//  Closes the underlying video write device
// --------------------------------------------------------------------
bool VidOStream::isOpen()
{
    return m_ostr;
}

// --------------------------------------------------------------------
//  Closes the underlying video read device
// --------------------------------------------------------------------
void VidIStream::pull()
{
}

// --------------------------------------------------------------------
//  Pushes a frame to the output stream
// --------------------------------------------------------------------
void VidOStream::push(Bitmap const& frame)
{
    if (!m_device) throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY,
        "Video stream must be open to push", Error_BadStream);

    m_device->addFrame(frame);
}

// --------------------------------------------------------------------
//  Acquires a video reader from a codec matching the format
// --------------------------------------------------------------------
void VidIStream::getReader(String const& format)
{
    // Acquire a read-lock on the modules for thread safe removal support
    boost::unique_lock<decltype(filescope::decodeMutex)> lock(filescope::decodeMutex);

    // Verify the video format has a registered encoder
    auto & module = filescope::decoders.find(format);
    if (module == filescope::decoders.end())
    {
        throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY, 
                    "No Video Decoder found", Error_BadToken);
    }

    // Acquire the resource streambuffer from the retriever
    m_device = module->second->reader();
}

// --------------------------------------------------------------------
//  Acquires a video writer from a codec matching the format
// --------------------------------------------------------------------
void VidOStream::getWriter(String const& format)
{
    // Acquire a read-lock on the modules for thread safe removal support
    boost::unique_lock<decltype(filescope::encodeMutex)> lock(filescope::encodeMutex);

    // Verify the video format has a registered encoder
    auto & module = filescope::encoders.find(format);
    if (module == filescope::encoders.end())
    {
        throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY, 
                    "No Video Encoder found", Error_BadToken);
    }

    // Acquire the resource streambuffer from the retriever
    m_device = module->second->writer();
}

// --------------------------------------------------------------------
//  Registers a new resource encoder module 
// --------------------------------------------------------------------
void VidIStream::registerDecoder(String const& format, std::shared_ptr<VideoDecoder> module)
{ 
    if (!module) throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY, 
        "registerDecoder requires valid handle", vox::Error_Range);

    // Acquire a read-lock on the modules for thread safe removal support
    boost::unique_lock<decltype(filescope::decodeMutex)> lock(filescope::decodeMutex);

    filescope::decoders[format] = module; 
}

// --------------------------------------------------------------------
//  Registers a new resource decoder module 
// --------------------------------------------------------------------
void VidOStream::registerEncoder(String const& format, std::shared_ptr<VideoEncoder> module)
{ 
    if (!module) throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY, 
        "registerEncoder requires valid handle", vox::Error_Range);

    // Acquire a read-lock on the modules for thread safe removal support
    boost::unique_lock<decltype(filescope::encodeMutex)> lock(filescope::encodeMutex);

    filescope::encoders[format] = module; 
}

// --------------------------------------------------------------------
//  Removes the resource module for a specified format
// --------------------------------------------------------------------
void VidIStream::removeDecoder(String const& format, std::shared_ptr<VideoDecoder> decoder)
{
    // Acquire a write-lock on the modules for thread safe removal support
    boost::unique_lock<decltype(filescope::decodeMutex)> lock(filescope::decodeMutex);

    auto iter = filescope::decoders.find(format);
    if (iter != filescope::decoders.end())
    {
        if (!decoder || iter->second == decoder) filescope::decoders.erase(format);
    }
}

// --------------------------------------------------------------------
//  Removes the resource module for a specified format
// --------------------------------------------------------------------
void VidOStream::removeEncoder(String const& format, std::shared_ptr<VideoEncoder> encoder)
{
    // Acquire a write-lock on the modules for thread safe removal support
    boost::unique_lock<decltype(filescope::encodeMutex)> lock(filescope::encodeMutex);

    auto iter = filescope::encoders.find(format);
    if (iter != filescope::encoders.end())
    {
        if (!encoder || iter->second == encoder) filescope::encoders.erase(format);
    }
}

// --------------------------------------------------------------------
//  Removes all registered instances of the specified IO Module
// --------------------------------------------------------------------
void VidIStream::removeDecoder(std::shared_ptr<VideoDecoder> decoder)
{
    // Acquire a write-lock on the modules for thread safe removal support
    boost::unique_lock<decltype(filescope::decodeMutex)> lock(filescope::decodeMutex);

    auto iter = filescope::decoders.begin();
    while (iter != filescope::decoders.end())
    {
        if (iter->second == decoder)
        {
            auto old = iter; ++iter;
            filescope::decoders.erase(old);
        }
        else
        {
            ++iter;
        }
    }
}

// --------------------------------------------------------------------
//  Removes all registered instances of the specified IO Module
// --------------------------------------------------------------------
void VidOStream::removeEncoder(std::shared_ptr<VideoEncoder> encoder)
{
    // Acquire a write-lock on the modules for thread safe removal support
    boost::unique_lock<decltype(filescope::encodeMutex)> lock(filescope::encodeMutex);

    auto iter = filescope::encoders.begin();
    while (iter != filescope::encoders.end())
    {
        if (iter->second == encoder)
        {
            auto old = iter; ++iter;
            filescope::encoders.erase(old);
        }
        else
        {
            ++iter;
        }
    }
}

} // namespace vox