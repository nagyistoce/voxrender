# Contents #



# Video IO #

**File:** https://code.google.com/p/voxrender/source/browse/trunk/Source/VoxLib/Video/VidStream.h_

Video IO in VoxLib is provided through the VideoStream classes. These functions take as input an OptionSet specifying module specific options as well as a ResourceID specifying the source/sink location.

# Video Formatting #

Videos are handled as a stream formatted bytes corresponding with a VideoCodec. This allows the import or export of videos to common formats such as mpeg or avi using the best compression algorithms available.

# Adding Video IO Modules #

# Example Usage #
```
    // Register the necessary IO modules
    auto & pm = PluginManager::instance();
    pm.loadFromFile("~/FileIO.dll");      // File IO protocols
    pm.loadFromFile("~/StandardImg.dll"); // Image Codecs
    pm.loadFromFile("~/StandardVid.dll"); // LibAV video codecs

    // Establish the URI of the resource
    ResourceId identifier = "file:///C:/my/path/video.mpeg";

    // Import some images
    try
    {
        VideoIStream vidStream(identifier);


    }
    catch(Error & error) { vox::Logger::addEntry(error); }
```