/* ===========================================================================

	Project: VoxRender - Debug Macros

	Description:
	 Defines some debug macros to be embedded in VOX_DEBUG flagged builds.

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
#ifndef VOX_DEBUG_H
#define VOX_DEBUG_H

// C++ Standard IO
#include <iostream>

// Debug break definition
#ifdef _MSC_VER
#  define DEBUG_BREAK __debugbreak( )
#else 
#  define DEBUG_BREAK
#endif

// Debug definitions
#ifdef VOX_DEBUG

	// Assertion failure message
	#define FAIL_MSG "Press Enter to Continue..."

	// Pause / Wait key
	#define VOX_PAUSE(msg) { std::cout << msg; std::cin.get( ); }

	// Break into debugger
	#define VOX_BREAK( ) { DEBUG_BREAK; }

	// Verify / Force eval
	#define VOX_VERIFY(expr) {							\
		if(!(expr)) {									\
			std::cout << "Verify fails on line "		\
			<< __LINE__ << " with expr \"" << #expr		\
			<< "\"\n"; VOX_BREAK(); } }

	// Assertion
	#define VOX_ASSERT(expr) {							\
		if(!(expr)) {									\
			std::cout << "Assertion fails on line "		\
			<< __LINE__ << " with expr \"" << #expr		\
			<< "\"\n"; VOX_PAUSE(FAIL_MSG); } }

	// Assertion with msg
	#define VOX_ASSERT_MSG(expr,msg) {	\
		if(!(expr)) {					\
			std::cout << (msg) << "\n";	\
			VOX_PAUSE(FAIL_MSG); } }

	// Trap / Halt Execution
	#define VOX_TRAP( ) {							\
			std::cout << "Trap triggered on line "	\
				<< __LINE__ << "\n";				\
			VOX_PAUSE(FAIL_MSG); }

	// Conditional
	#define VOX_IF_DBG(stmt) { stmt }

#else // VOX_DEBUG

	#define VOX_PAUSE(msg)
	#define VOX_BREAK( )
	#define VOX_VERIFY(expr) {expr;}
	#define VOX_ASSERT(expr)
	#define VOX_ASSERT_MSG(expr,msg)
	#define VOX_TRAP( )
	#define VOX_IF_DBG(stmt)

#endif // VOX_DEBUG

// End definition
#endif // VOX_DEBUG_H