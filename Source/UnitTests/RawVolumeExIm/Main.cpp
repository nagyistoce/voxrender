/* ===========================================================================

    Project: VoxRender - VoxLib Raw Volume Test Module

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

#define BOOST_TEST_MODULE "Vox Raw Volume ExIm Test Module"

// Include Boost UnitTest Framework
#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>

// Include Dependencies
#include "VoxLib/Core/Functors.h"
#include "VoxLib/Core/Logging.h"
#include "VoxLib/IO/Resource.h"
#include "VoxLib/IO/ResourceId.h"
#include "VoxScene/Scene.h"
#include "VoxScene/Volume.h"
#include "Plugins/FileIO/FileIO.h"
#include "Plugins/RawVolumeImporter/RawVolumeImporter.h"

#include <stdlib.h>     // srand, rand 
#include <time.h>       // time 

using namespace vox;

// --------------------------------------------------------------------
//  Tests the raw volume export/import given a volume and option set
// --------------------------------------------------------------------
void testVolumeExIm(Scene const& scene, String const& compression)
{
    std::cout << "Testing compression mode \"" << compression << "\"" << std::endl;

    String const identifier = "file:///" + boost::filesystem::current_path().string() + "/test_" + compression + ".raw";

    // Construct the option set
    OptionSet options;
    options.addOption("Compression", compression);

    // Export the scene information
    ResourceOStream out(identifier);
    scene.exprt(out, options); 
    out.close();

    // Import scene information
    options.addOption("Type", scene.volume->typeToString(scene.volume->type()));
    options.addOption("Size", "[256 256 256 1]");
    Scene copy = Scene::imprt(identifier, options);

    // Ensure the volume data was imported sucessfully
    auto const* dataOrig = scene.volume->data();
    auto const* dataCopy = copy.volume->data();
    size_t bytes = 256*256*256*1*1;
    BOOST_CHECK( memcmp(dataOrig, dataCopy, bytes) == 0 );

    // Remove the temporary files
    Resource::remove(identifier);
}

// --------------------------------------------------------------------
//  Generates a randomized volume test data set
// --------------------------------------------------------------------
std::shared_ptr<UInt8> generateTestData(size_t bytes)
{
    auto data = makeSharedArray(bytes);

    auto ptr = data.get();
    for (size_t i = 0; i < bytes; i++)
        *ptr++ = rand() % 256;

    return data;
}

// --------------------------------------------------------------------
//  Performs tests of the Plugin load and unload functionality
// --------------------------------------------------------------------
BOOST_AUTO_TEST_SUITE( RawVolumeExIm )

    // Tests a generated data set
    BOOST_AUTO_TEST_CASE( DataCompression )
    {
        srand(time(nullptr));

        std::cout << "Testing raw volume ExIm compression modes" << std::endl;

        // Register the raw volume file ExIm and a filesystem IO module
        auto exim = std::shared_ptr<RawVolumeFile>(new RawVolumeFile(nullptr));
        Scene::registerImportModule(".raw", exim);
        Scene::registerExportModule(".raw", exim);
        Resource::registerModule("file", std::shared_ptr<FileIO>(new FileIO));
     
        // Specify the test volume parameters
        Vector4u extent(256, 256, 256, 1);
        Vector4f spacing(1.0f, 1.0f, 1.0f, 1.0f);
        size_t   bpv     = 1;
        size_t   nVoxels = 256*256*256*1*1;

        // Generate a random test volume data set
        auto data   = generateTestData(nVoxels);
        auto volume = Volume::create(data, extent, spacing);

        // Construct scene object for export
        Scene scene; scene.volume = volume;

        // Export some raw volume files
        testVolumeExIm(scene, "");
        testVolumeExIm(scene, "zlib");
        testVolumeExIm(scene, "bzip2");
        testVolumeExIm(scene, "gzip");
        testVolumeExIm(scene, "zlib bzip2");
        testVolumeExIm(scene, "bzip2 zlib");
        testVolumeExIm(scene, "gzip bzip2");
        testVolumeExIm(scene, "bzip2 gzip");
        testVolumeExIm(scene, "gzip zlib");
        testVolumeExIm(scene, "zlib gzip");
        testVolumeExIm(scene, "zlib bzip2 gzip");
        testVolumeExIm(scene, "gzip zlib bzip2");
    }
    
BOOST_AUTO_TEST_SUITE_END()