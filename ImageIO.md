# Contents #



# Image IO #

**File:** [\_VoxLib/Bitmap/Bitmap.h\_](https://code.google.com/p/voxrender/source/browse/trunk/Source/VoxLib/Image/RawImage.h)

Image IO in VoxLib is provided through the Bitmap class functions exprt() and imprt(). These functions take as input an OptionSet specifying module specific options as well as a ResourceID specifying the source/sink location. In addition to the ResourceID functions, there are overloads for providing ResourceStream objects directly to the import/export modules.

# Image Formatting #

Images are handled as a formatted array of bytes with a given format, bit depth, and stride. This allows the import or export of images to common formats such as JPEG or PNG using the best compression algorithms available. If the type of the image does not match one of the provided enums in Bitmap::Format, then the Format\_Unknown type can be used to indicate that the image should be encoded without assuming knowledge of the underlying data's meaning.

# Adding Image IO Modules #

The VoxIO library is used to open, close, and manage the underlying image data. The codecs for handling the processing of the image data however are registered with the RawImage class functions registerExModule() and registerImModule().

When a module is registered, it is mapped to a user specified file extension. When an import or export call is made using the image library, the extension of the ResourceID parameter will be analyzed and matched to a given codec. The ResourceID extension can be forced to a different string using an overload with an additional extension parameter. This is useful for performing encoding/decoding in memory.

This functions perform image encoding and decoding respectively. The included module (StandardImg) partially supports the following image formats as of now:

  * PNG
  * JPG
  * BMP

# Example Usage #
```
    // Register the necessary IO modules
    auto & pm = PluginManager::instance();
    pm.loadFromFile("~/FileIO.dll");      // File IO protocols
    pm.loadFromFile("~/StandardImg.dll"); // JPEG Image Codecs
    pm.loadFromFile("~/StandardIO.dll");  // Network IO protocols

    // Establish the URI of the resource
    ResourceId identifier = "file:///C:/my/path/image.png";

    // Import some images
    try
    {
        auto image1 = Bitmap::imprt(identifier);
        auto image2 = Bitmap::imprt("http://example.org/exampleImg.jpg"); 

        // Perform processing of images //
    }
    catch(Error & error) { vox::Logger::addEntry(error); }
```