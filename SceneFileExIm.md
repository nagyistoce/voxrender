# Scene File Import/Export in VoxLib #

**File:** _VoxLib/Scene/Scene.h_

The Scene file import/export (referred to as imprt/exprt in the documentation as the naming convention is selected to avoid issues with C++ keywords) is managed by breaking down the process into 2 stages.

(1) The **ACCESS** stage involves a request being sent to the IO layer to acquire a c++ IO stream for a resource based on its URI. Modules for different access protocols are registered in the form of a call to _vox::Resource::registerModule_ with the string identifying the protocol which the input _vox::ResourceModule_ interface will handle. A FilesystemIO implementation is provided with the library in addition to a wrapper for the popular libcurl library. The libcurl wrapper provides ftp, sftp, http, and https support.

(2) The **PARSING** stage involves a registered handler for a given file type reading the stream object provided by the IO layer. Functions for different file extensions are registered in the form of calls to _vox::Scene::registerImportModule_ and _vox::Scene::registerExportModule_. A scene file import or export module may, as necessary, make recursive calls to the **ACCESS** stage to provide handles to resources specified in the input document. It is the responsibility of the module to ensure that any relative URIs are applied relative to the base URI of the initial stream it was given.
The library's usage of URIs for resource identification conforms to the RFC guidelines as noted in the documentation accompanying the library.

# Example Usage #
```
    // Register the scene file import / export modules
    vox::Scene::registerImportModule(".xml", &vox::VoxSceneFile::importer   );
    vox::Scene::registerImportModule(".raw", &vox::RawVolumeFile::importer  );
    vox::Scene::registerImportModule(".vtf", &vox::VoxTransferFile::importer);
    vox::Scene::registerExportModule(".xml", &vox::VoxSceneFile::exporter   );
    vox::Scene::registerExportModule(".raw", &vox::RawVolumeFile::exporter  );
    vox::Scene::registerExportModule(".vtf", &vox::VoxTransferFile::exporter);

    // Register the resource opener modules
    vox::Resource::registerModule("file", FilesystemIO::create());

    // Establish the URI of the scene file
    //
    // The IO library will automatically append a base URI
    // for the initial imprt or exprt URI which is an application
    // default of the file protocol on the local filesystem.
    //
    // While importing a scene file with the 'vox' XML file loader
    // however, relative URI references will be applied to the 
    // document's base URI in accordance with RFC guidelines.
    String const identifier = "file:///C:/my/path/file.xml";

    // Import the specified scene file
    try
    {
        auto scene = vox::Scene::imprt(identifier);

        // Perform processing of scene //
    }
    catch(Error & error) { vox::Logger::addEntry(error); }
```

# Import/Export Modules #

As shown in the code snippet above, the VoxLib library provides a few scene import/export modules. Modules are registered at runtime, permitting you to implement your own and deploy them internally or as a plugin using the PluginManager class.