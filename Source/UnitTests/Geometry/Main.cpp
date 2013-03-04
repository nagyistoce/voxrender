/* ===========================================================================

	Project: VoxRender - VoxIO Unit Test Module

	Description: Performs unit testing for the IO library features

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

#define BOOST_TEST_MODULE "Vox Geometry Unit Test Module"

// Include Boost UnitTest Framework
#include <boost/test/unit_test.hpp>

// Include Dependencies
#include "VoxLib/Core/Geometry.h"
#include "VoxLib/Core/Functors.h"

using namespace vox;

// --------------------------------------------------------------------
//  Performs tests of the ResourceId parsing and management functions
// --------------------------------------------------------------------
BOOST_AUTO_TEST_SUITE( Vector )

    // Tests the vector's construction
    BOOST_AUTO_TEST_CASE( Construction )
    {
        Vector3f vec1(0.0f, 1.0f, 2.0f);
        BOOST_CHECK(vec1[0] == 0.0f);
        BOOST_CHECK(vec1[1] == 1.0f);
        BOOST_CHECK(vec1[2] == 2.0f);
        
        Vector4 vec2(1, 3, 4, 5);
        BOOST_CHECK(vec2[0] == 1);
        BOOST_CHECK(vec2[1] == 3);
        BOOST_CHECK(vec2[2] == 4);
        BOOST_CHECK(vec2[3] == 5);
        vec2.assign(6, 7, 8, 9);
        BOOST_CHECK(vec2[0] == 6);
        BOOST_CHECK(vec2[1] == 7);
        BOOST_CHECK(vec2[2] == 8);
        BOOST_CHECK(vec2[3] == 9);

        vox::Vector<Vector2f,2> vec3(
            Vector2f(0.0f, 1.0f),
            Vector2f(3.0f, 4.0f));
        BOOST_CHECK(vec3[0][0] == 0.0f);
        BOOST_CHECK(vec3[0][1] == 1.0f);
        BOOST_CHECK(vec3[1][0] == 3.0f);
        BOOST_CHECK(vec3[1][1] == 4.0f);
    }

    // Tests the vector's basic fuctionality
    BOOST_AUTO_TEST_CASE( BasicOps )
    {
        {
        Vector3f vec1(0.0f, 2.0f, 0.0f);
        Vector3f vec2(1.0f, 4.0f, 1.0f);
        BOOST_CHECK(vec1-vec2 == Vector3f(-1.0f, -2.0, -1.0f));
        BOOST_CHECK(vec1+vec2 == Vector3f(1.0f, 6.0, 1.0f));
        BOOST_CHECK(Vector3f::dot(vec1, vec2) == 8.0f);
        }
    }

    // Some helper functions for testing map and fold
    size_t add2Functor(size_t const& x)  { return x + 2; }
    void   addAssign2Functor(size_t & x) { x += 2; }

    // Tests the vector's map and fold functionality
    BOOST_AUTO_TEST_CASE( VectorOps )
    {
        {
        Vector3u vec(3, 5, 4);
        vec.map(&add2Functor);
        BOOST_CHECK(vec[0] == 5);
        BOOST_CHECK(vec[1] == 7);
        BOOST_CHECK(vec[2] == 6);
        }
        
        {
        Vector3u vec(3, 5, 4);
        vec.map([] (size_t const& x) { return x + 2; });
        BOOST_CHECK(vec[0] == 5);
        BOOST_CHECK(vec[1] == 7);
        BOOST_CHECK(vec[2] == 6);
        }
        
        {
        Vector3u vec(8, 2, -1000);
        vec.mapAssign(&addAssign2Functor);
        BOOST_CHECK(vec[0] == 10);
        BOOST_CHECK(vec[1] == 4);
        BOOST_CHECK(vec[2] == -998);
        }

        {
        Vector3f vec(3.0f, 8.0f, 0.5f);
        BOOST_CHECK(vec.high() == 8.0f);
        BOOST_CHECK(vec.low()  == 0.5f);
        }
    }

    // Tests the vector's std container convention compliance
    BOOST_AUTO_TEST_CASE( ContainerSupport )
    {
        {
        Vector4u vec(1, 1, 1, 1);
        BOOST_FOREACH (auto & elem, vec)
        {
            BOOST_CHECK(elem == 1);
        }
        }

        {
        Vector4u vec(1, 1, 1, 1);
        BOOST_REVERSE_FOREACH (auto & elem, vec)
        {
            BOOST_CHECK(elem == 1);
        }
        }
    }

    // Tests the matrix functionality
    BOOST_AUTO_TEST_CASE( Matrix )
    {
        {
        Matrix3x3f vec1(Vector3f(0.0f, 2.0f, 0.0f),
                        Vector3f(0.0f, 2.0f, 1.0f),
                        Vector3f(5.0f, 3.0f, 8.0f));
        }
    }

BOOST_AUTO_TEST_SUITE_END()