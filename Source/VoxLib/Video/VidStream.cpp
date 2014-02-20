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