/* ===========================================================================

	Project: VoxRender - Format

	Description: 
        Overloads the boost::format interface for ease of use in library calls.

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
#ifndef VOX_FORMAT_H
#define VOX_FORMAT_H

// Max repeat macro depth
#ifndef VOX_FORMAT_LIMIT
#define VOX_FORMAT_LIMIT 10
#endif // VOX_FORMAT_LIMIT

// Include Dependencies
#include "VoxLib/Core/CudaCommon.h"
#include "VoxLib/Core/Preprocessor.h"

// API Namespace
namespace vox
{

// Modified formatting interface for readability
#define VOX_FORMAT_CONSTRUCTOR(z, n, _)                     \
    template< BOOST_PP_ENUM_PARAMS_Z(z, n, class T) >       \
    std::string format( char const* string,                 \
        BOOST_PP_ENUM_BINARY_PARAMS_Z(z, n, T, const& a) )  \
    {                                                       \
        return (boost::format( string )                     \
            PP_ENUM_PARAMS_NO_COMMA_Z(z, n, % a)            \
            ).str( );                                       \
    }                                                       

    BOOST_PP_REPEAT_FROM_TO( 1, VOX_FORMAT_LIMIT, VOX_FORMAT_CONSTRUCTOR, _)

#undef VOX_FORMAT_CONSTRUCTOR

    template<typename T>    
    std::string format( T const& a ) 
    { 
        return str( boost::format("%1%") % a ); 
    } 

}

// End definition
#endif // VOX_FORMAT_H