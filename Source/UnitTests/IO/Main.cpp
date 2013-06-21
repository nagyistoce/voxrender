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

#define BOOST_TEST_MODULE "Vox IO Unit Test Module"

// Include Boost UnitTest Framework
#include <boost/test/unit_test.hpp>

// Include Dependencies
#include "VoxLib/Core/System.h"
#include "VoxLib/Error/Error.h"
#include "VoxLib/IO/MimeTypes.h"
#include "VoxLib/IO/Resource.h"
#include "VoxLib/IO/ResourceId.h"
#include "Plugins/FileIO/FileIO.h"

using namespace vox;

// --------------------------------------------------------------------
//  Performs tests of the ResourceId parsing and management functions
// --------------------------------------------------------------------
BOOST_AUTO_TEST_SUITE( ResourceIdSuite )

    // Tests the URI parsing functionality
    BOOST_AUTO_TEST_CASE( Parsing_Test )
    {
        {
        ResourceId uri("ftp://username@hostname:0/path/to/file.txt?this=is&query=string#fragment_id");
        BOOST_CHECK( uri.extractHostname()      == "hostname" );
        BOOST_CHECK( uri.extractFileName()      == "file.txt" );
        BOOST_CHECK( uri.extractFileExtension() == ".txt"     );
        BOOST_CHECK( uri.extractPortNumber()    == 0          );
        BOOST_CHECK( uri.extractUserinfo()      == "username" );
        BOOST_CHECK( uri.query    == "this=is&query=string" );
        BOOST_CHECK( uri.fragment == "fragment_id"          );
        BOOST_CHECK( uri.path     == "/path/to/file.txt"    );
        BOOST_CHECK( uri.scheme   == "ftp"                  );
        }
        
        {
        ResourceId uri("HTTP://MyUserName:Field@HoSt%20NAme:40?this=is%20&%20query=string");
        BOOST_CHECK( uri.extractHostname()           == "HoSt NAme" );
        BOOST_CHECK( uri.extractNormalizedHostname() == "host name" );
        BOOST_CHECK( uri.extractPortNumber()         == 40);
        BOOST_CHECK( uri.extractUserinfo()           == "MyUserName:Field" );
        BOOST_CHECK( uri.query  == "this=is & query=string" );
        BOOST_CHECK( uri.scheme == "http" );
        BOOST_CHECK( uri.fragment.empty() );
        BOOST_CHECK( uri.path.empty());
        }

        // :TODO: test '+' symbols

        {
        BOOST_CHECK_THROW(ResourceId uri("http://hostname/path%2"), Error);
        BOOST_CHECK_THROW(ResourceId uri("http://hostname/path%2."), Error);
        }
    }

    // Tests the query param functionality
    BOOST_AUTO_TEST_CASE( Query_Params_Test )
    {
        {
        ResourceId uri("http://host/?string=abc&int=5&float=1.0");
        OptionSet options = uri.extractQueryParams();
        BOOST_CHECK( options.lookup("string") == "abc" );
        BOOST_CHECK( options.lookup<float>("float") == 1.0f );
        BOOST_CHECK( options.lookup<int>("int") == 5 );
        }

        {
        ResourceId uri("http://host/?string=abc&int=5&float=1.4f");
        OptionSet options;
        options.addOption("string", "1&2=3 ");
        options.addOption("int", 2);
        options.addOption("double", 0.0);
        uri.setQueryParams(options);
        options = uri.extractQueryParams();
        BOOST_CHECK( options.lookup("string") == "1&2=3 " );
        BOOST_CHECK( options.lookup<float>("double") == 0.0 );
        BOOST_CHECK( options.lookup<int>("int") == 2 );
        }

        {
        ResourceId uri("http://host/?string=abc&int=5&float=1.4f");
        OptionSet options = uri.extractQueryParams();
        BOOST_CHECK_THROW(options.lookup("no-exist"), Error);
        BOOST_CHECK( options.lookup("no-exist","val") == "val" );
        BOOST_CHECK( options.lookup("no-exist",5) == 5 );
        BOOST_CHECK_THROW(options.lookup("string",5), Error);
        }
    }
    
BOOST_AUTO_TEST_SUITE_END()
    
// --------------------------------------------------------------------
//  Performs tests of the resource module registration and usage
// --------------------------------------------------------------------
BOOST_AUTO_TEST_SUITE( ResourceModulesTest )

    // Tests the built-in filesystem IO library
    BOOST_AUTO_TEST_CASE( FilesystemIOTest )
    {
        String const data = "abcdefghijklmnopqrstuvwxyz\n!@#$%^&*()1234567890\r\n";
   
        String const identifier = "file:///" + System::currentDirectory() + "/test_FileIO.txt";

        // Register the FilesystemIO resource module
        Resource::registerModule("file", std::shared_ptr<FileIO>(new FileIO()));
     
        // Stream some output data to a file
        ResourceOStream out(identifier);
        out.write(&data[0], data.size());
        BOOST_CHECK(out.good());
        out.close();

        // Read in the streamed data
        ResourceIStream in(identifier);
        String readData(data.size(), ' '); 
        in.read(&readData[0], data.size());
        BOOST_CHECK(in.good());
        in.close();

        // Check the data for errors
        BOOST_CHECK( memcmp(readData.c_str(), data.c_str(), data.size()) == 0 );

        // Append some more data
        out.open(identifier, Resource::Mode_Append);
        out.write(&data[0], data.size());
        BOOST_CHECK(out.good());
        out.close();

        // Compute new file content
        String newData = data + data;

        // Read in the streamed data
        in.open(identifier);
        String newReadData(newData.size(), ' '); 
        in.read(&newReadData[0], newData.size());
        BOOST_CHECK(in.good());
        in.close();

        // Check the data for errors
        BOOST_CHECK( memcmp(newReadData.c_str(), newData.c_str(), newData.size()) == 0 );
    }
    
BOOST_AUTO_TEST_SUITE_END()
