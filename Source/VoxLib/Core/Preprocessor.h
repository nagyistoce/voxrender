/* ===========================================================================

	Project: VoxRender - Preprocessor macros

	Includes the boost preprocessor macros

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
#ifndef VOX_PREPROCESSOR_H
#define VOX_PREPROCESSOR_H

// Boost Preprocessor Headers
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/enum_params.hpp>
#include <boost/preprocessor/punctuation.hpp>
#include <boost/preprocessor/repetition/enum.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/repetition/repeat_from_to.hpp>
#include <boost/preprocessor/stringize.hpp>

// Array initialization from enum parameters
#define PP_ARRAY_ASSIGN_ELEM(z, n, data)                \
    BOOST_PP_CAT(BOOST_PP_SEQ_ELEM(0, data), [n] =)     \
    BOOST_PP_CAT(BOOST_PP_SEQ_ELEM(1, data), n;) 

#define PP_ARRAY_ASSIGN_FROM_TO(z, start, end, data)    \
    BOOST_PP_REPEAT_FROM_TO(start, end,                 \
        PP_ARRAY_ASSIGN_ELEM, data)

#define PP_ARRAY_ASSIGN(z, n, data)         \
    BOOST_PP_REPEAT_FROM_TO(0, n,           \
        PP_ARRAY_ASSIGN_ELEM, data)

// Enum parameters macro which does not enforce comma delimination
#define PP_NO_COMMA_PARAM(z, n, data) BOOST_PP_CAT(data, n)

#define PP_ENUM_PARAMS_NO_COMMA_Z(z, count, param)           \
    BOOST_PP_REPEAT(count, PP_NO_COMMA_PARAM, param)

// End definition
#endif // VOX_PREPROCESSOR_H