/* ===========================================================================

	Project: VoxRender

	Description: Performs unit testing for the VoxServer library

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

#define BOOST_TEST_MODULE "VoxServer Unit Test Module"

// Include Boost UnitTest Framework
#include <boost/test/unit_test.hpp>

// Convenience Header for Server DLL
#include "VoxServer/Interface.h"

using namespace vox;

// --------------------------------------------------------------------
//  Performs tests of the ResourceId parsing and management functions
// --------------------------------------------------------------------
BOOST_AUTO_TEST_SUITE( Server )

    // Tests the matrix functionality
    BOOST_AUTO_TEST_CASE( WebSocketConnect )
    {
        while (true)
        {
            voxServerStart("C:/Users/Lucas/Documents/Projects/voxrender/trunk/Binaries/x86/Debug", false);

            UInt16 port = 0;
            UInt64 key  = 0;
            voxServerBeginStream(&port, &key, "file:///C:/Users/Lucas/Documents/Projects/voxrender/trunk/Models/Examples/");

            voxServerEnd();
        }
    }

BOOST_AUTO_TEST_SUITE_END()