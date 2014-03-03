/* ===========================================================================

	Project: VoxRender - Types

	Description: Typedefs for library usage

    Copyright (C) 2012 Lucas Sherman

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
#ifndef VOX_TYPES_H
#define VOX_TYPES_H

// Pull in std::string file headers
#include "VoxLib/Core/CudaCommon.h"

// Use Boost cstdint for MSVC 9 compatibility
#include <boost/cstdint.hpp>

// API namespace
namespace vox
{
    // Standard width types
    typedef boost::uint8_t  UInt8;
    typedef boost::uint16_t UInt16;
    typedef boost::uint32_t UInt32;
    typedef boost::uint64_t UInt64;
    typedef boost::int8_t   Int8;
    typedef boost::int16_t  Int16;
    typedef boost::int32_t  Int32;
    typedef boost::int64_t  Int64;

    // Determine primitive type for desired encoding
    typedef char Char;

    // Standard library - encoding specific classes
    typedef std::basic_string<Char>        String;
    typedef std::basic_stringstream<Char>  StringStream;
    typedef std::basic_istringstream<Char> IStringStream;
    typedef std::basic_ostringstream<Char> OStringStream;
    typedef std::basic_stringbuf<Char>     StringBuf;
    typedef std::basic_fstream<Char>       FileStream;
    typedef std::basic_ifstream<Char>      IFileStream;
    typedef std::basic_ofstream<Char>      OFileStream;
    typedef std::basic_filebuf<Char>       FileBuf;
    typedef std::basic_iostream<Char>      IOStream;
    typedef std::basic_istream<Char>       IStream;
    typedef std::basic_ostream<Char>       OStream;
    typedef std::basic_ios<Char>           IOS;

    // String format conversion functionality
    // :TODO:
}

// End definition
#endif // VOX_TYPES_H