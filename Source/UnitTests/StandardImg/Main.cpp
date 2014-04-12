/* ===========================================================================

    Project: VoxRender - StandardImg Plugin Test Module

    Description: Performs unit testing for the Standard IO plugin

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

#define BOOST_TEST_MODULE "Vox StandardIO Unit Test Module"

// Include Boost UnitTest Framework
#include <boost/test/unit_test.hpp>

// Include Dependencies
#include "VoxLib/Core/Types.h"
#include "VoxLib/Core/Functors.h"
#include "VoxLib/Core/Logging.h"
#include "VoxLib/Error/Error.h"
#include "VoxLib/Plugin/Plugin.h"
#include "VoxLib/IO/Resource.h"
#include "VoxLib/IO/ResourceHelper.h"
#include "VoxLib/Plugin/PluginManager.h"
#include "VoxLib/Bitmap/Bitmap.h"

#include <boost/thread.hpp>
#include <boost/date_time.hpp>
#include <boost/filesystem.hpp>

using namespace vox;

// --------------------------------------------------------------------
//  Performs tests of the Plugin load and unload functionality
// --------------------------------------------------------------------
BOOST_AUTO_TEST_SUITE( StandardImgSuite )

    // Tests the plugins meta info functions
    BOOST_AUTO_TEST_CASE( MetaInfoTest )
    {
    }
    
    // Tests the plugins PNG load functionality
    BOOST_AUTO_TEST_CASE( PNGTest )
    {
        size_t const N_PATTERNS = 10;
        String const IDENTIFIER = "file:///" + boost::filesystem::current_path().generic_string() + "/test.png";

        // Load the vox.standard_io plugin
        auto & pluginManager = PluginManager::instance();
        pluginManager.loadFromFile("StandardImg.dll");
        pluginManager.loadFromFile("FileIO.dll");

        // RGB test patterns
        for (size_t i = 0; i < N_PATTERNS; i++)
        {
            auto w = (size_t)(rand() % 1024);
            auto h = (size_t)(rand() % 1024);
            auto b = w * 4 + (16 - ((w*4)%16));
            auto s = b * h;
            auto d = makeSharedArray(s);

            auto * ptr = d.get();
            for (size_t x = 0; x < w; x++)
            for (size_t y = 0; y < h; y++)
                ((UInt32*)(ptr + b*y))[x] = 0xFF0000FF;

            Bitmap imageO(Bitmap::Format_RGBA, w, h, 8, 1, b, d);
            imageO.exprt(IDENTIFIER);
            auto imageI = Bitmap::imprt(IDENTIFIER);

            auto optr = (char*)imageO.data();
            auto iptr = (char*)imageI.data();
            for (size_t i = 0; i < imageI.height(); i++)
            {
                BOOST_CHECK(memcmp(optr, iptr, s) == 0);
                optr += imageO.stride();
                iptr += imageI.stride();
            }
        }

        pluginManager.unloadAll();
    }

BOOST_AUTO_TEST_SUITE_END()
