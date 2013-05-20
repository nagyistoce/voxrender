/* ===========================================================================

    Project: VoxRender - VoxIO Plugin Test Module

    Description: Performs unit testing for plugin management

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

#define BOOST_TEST_MODULE "Vox StandardIO Unit Test Module"

// Include Boost UnitTest Framework
#include <boost/test/unit_test.hpp>

// Include Dependencies
#include "VoxLib/Core/Types.h"
#include "VoxLib/Error/Error.h"
#include "VoxLib/Plugin/Plugin.h"
#include "VoxLib/IO/FilesystemIO.h"
#include "VoxLib/IO/Resource.h"

using namespace vox;

// --------------------------------------------------------------------
//  Performs tests of the Plugin load and unload functionality
// --------------------------------------------------------------------
BOOST_AUTO_TEST_SUITE( StandardIOSuite )

    // Tests the plugin load/unload functions
    BOOST_AUTO_TEST_CASE( MetaInfoTest )
    {
        std::fstream infile("C:/Users/lucas/Documents/Projects/voxrender/wiki/images/logo.bmp", std::ios::binary|std::ios::in);

        UInt16 header; infile.read((char*)&header, sizeof(UInt16)); std::cout << header << std::endl;
        UInt32 size;   infile.read((char*)&size,   sizeof(UInt32)); std::cout << size   << std::endl;
        UInt32 app;    infile.read((char*)&app,    sizeof(UInt32)); std::cout << app    << std::endl;
        UInt32 begin;  infile.read((char*)&begin,  sizeof(UInt32)); std::cout << begin  << std::endl;

        infile.ignore(begin - 4 - 4 - 4 - 2);

        size_t const bwidth  = 512;
        size_t const bheight = 128;
        UInt8 * bitmap = new UInt8[bwidth*bheight*3];
        infile.read((char*)bitmap, bwidth*bheight*3);
        for (size_t i = 0; i < bwidth*bheight; i++)
        {
            std::cout << ((bitmap[i*3]==0) ? 1 : 0);
        }

        size_t const width  = 512;
        size_t const height = 128+64;
        size_t const depth  = 128;
        UInt8 * data = new UInt8[width*height*depth];
        memset(data, 0, width*height*depth);

        // Build base for lettering
        for (size_t x = 0; x < 512; x++)
        for (size_t y = 0; y < 64; y++)
        for (size_t z = 0; z < 128; z++)
        {
            size_t i = x + y * width + z * width * height;
            data[i] = 128;
        }

        // Build lettering
        for (size_t x = 0; x < 512; x++)
        for (size_t y = 64; y < 128+64; y++)
        for (size_t z = 40; z < 88; z++)
        {
            size_t iv = x + y * width + z * width * height;
            size_t ib = x + (y-64) * bwidth;

            data[iv] = ((bitmap[ib*3]==0) ? 196 : 0);

            //std::cout << ((bitmap[ib*3]==0) ? 'X' : ' ');
        }

        std::fstream outfile("C:/Users/lucas/Documents/Projects/voxrender/trunk/Models/Logo/logo.raw", std::ios::binary|std::ios::out);

        outfile.write((char*)data, width*depth*height);

        delete[] bitmap;
        delete[] data;
    }
    
BOOST_AUTO_TEST_SUITE_END()