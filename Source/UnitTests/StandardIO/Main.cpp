/* ===========================================================================

    Project: VoxRender - StandardIO Plugin Test Module

    Description: Performs unit testing for the Standard IO plugin

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
#include "VoxLib/Core/Functors.h"
#include "VoxLib/Core/Logging.h"
#include "VoxLib/Error/Error.h"
#include "VoxLib/Plugin/Plugin.h"
#include "VoxLib/IO/FilesystemIO.h"
#include "VoxLib/IO/Resource.h"

// Standard IO Header
#include "Plugins/StandardIO/StandardIO.h"

#include <boost/thread.hpp>
#include <boost/date_time.hpp>

using namespace vox;

// --------------------------------------------------------------------
//  Performs tests of the Plugin load and unload functionality
// --------------------------------------------------------------------
BOOST_AUTO_TEST_SUITE( StandardIOSuite )

    // Tests the plugins meta info functions
    BOOST_AUTO_TEST_CASE( MetaInfoTest )
    {
    }
    
    // Tests the plugins HTTP IO functionality
    BOOST_AUTO_TEST_CASE( HttpIOTest )
    {
        std::cout << "**** This test requires an external network connection. ****" << std::endl;

        vox::Logger::setFilteringLevel(Severity_Trace);

        auto io = std::shared_ptr<StandardIO>(new StandardIO());

        // LibCurl supported protocols suite
        Resource::registerModule("http",   io);
        Resource::registerModule("https",  io);
        Resource::registerModule("ftp",    io);
        Resource::registerModule("ftps",   io);
        Resource::registerModule("sftp",   io);
        Resource::registerModule("tftp",   io);
        Resource::registerModule("rtmp",   io);
        Resource::registerModule("rtsp",   io);
        Resource::registerModule("smtp",   io);
        Resource::registerModule("smtps",  io);
        Resource::registerModule("dict",   io);
        Resource::registerModule("scp",    io);
        Resource::registerModule("imap",   io);
        Resource::registerModule("imaps",  io);
        Resource::registerModule("pop3",   io);
        Resource::registerModule("pop3s",  io);
        Resource::registerModule("ldap",   io);
        Resource::registerModule("ldaps",  io);
        Resource::registerModule("gopher", io);
        Resource::registerModule("telnet", io);

        // ftp://ftp.funet.fi/README
        // http://www.example.com
        // dict://dict.org/m:curl
        // gopher://gopher.quux.org:70/About%20This%20Server.txt

        ResourceId example("imaps://LucasASherman@imap.gmail.com");
        ResourceIStream webpageStream(example);

        //std::ofstream os(example.extractFileName()); os << webpageStream.rdbuf();

        // Logging to console using rdbuf locks the output stream (used for internal logging)
        OStringStream os; os << webpageStream.rdbuf();
        std::cout << os.str();
    }

BOOST_AUTO_TEST_SUITE_END()
