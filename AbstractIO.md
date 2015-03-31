# Contents #



# Resource Identification #

**File:** [\_VoxLib/IO/ResourceId.h\_](https://code.google.com/p/voxrender/source/browse/trunk/Source/VoxLib/IO/ResourceId.h)

The abstract IO library in VoxLib uses uniform resource identifiers (URIs) to identify and access resources. The URIs are managed with a convenience structure 'ResourceId' which provides read/write access to the various components of a URI.

The generic syntax for a uniform resource identifier as it pertains to this interface follows the standards outlined in RFC-3986 and RFC-3987 (http://tools.ietf.org/html/rfc3986) and http://tools.ietf.org/html/rfc3987) which provides this basic outline:

```
URI = <scheme name> :// <authority> <path> [ ? <query> ] [ # <fragment> ]
```

Within VoxLib, the `[ <userinfo> @ ] <host> [ : <port> ]` portion of the `<authority>` of the URI is handled as a single hierarchical component internally to facilitate variations in schemes. As such, retrieval modules are responsible for the parsing and verification of the URI `<authority>` component.

Detections of violations of this format during management of the URI by any part of the abstract IO interface will and should result in the rejection of the URI for the current operation (Typically indicated by throwing an exception) rather than a silent failure or ignorance of the violating characters.

For more detailed information on the ResourceId structure and the adherence to RFC guidelines, see the VoxLib documentation for 'ResourceId.h'. This documentation provides an in depth description of how the URI strings are managed along with information on normalization, query strings, and percent encoding.

# Resource Management #

**File:** [\_VoxLib/IO/Resource.h\_](https://code.google.com/p/voxrender/source/browse/trunk/Source/VoxLib/IO/ResourceId.h)

There are three distinct operations that can be performed on a resource through the Resource class:

## (1) Resource Access ##

Resources are accessed and modified through ResourceStream objects which mirror and inherit from their std::iostream counterparts:

  * Resource (std::ios)
  * ResourceOStream (std::ostream)
  * ResourceIStream (std::istream)
  * ResourceStream (std::iostream)

std::ios mode flags are however ignored in favor of Resource::Mode flags, which provide more explicit specifications of behavior. The exact specifications of these flags can be found in the source documentation.

## (2) Resource Removal ##

Resources can be removed or deleted through the function _Resource::remove_. A successful call to this function should delete a resource from the target URI based on the convention for the URI scheme.

## (3) Resource Query ##

Resources can be queried through the function _Resource::query_. A resource query returns a property\_tree containing information about the resource that is provided at the discretion of the supplying module. The library provides no restrictions beyond this and different modules may provide different results. As such, if you permit the use of plugins or other unpredictable module replacement tools, you should be prepared to dynamically asses what information is available in a returned tree.

# Mime Types Support #

**File:** [\_VoxLib/IO/MimeTypes.h\_](https://code.google.com/p/voxrender/source/browse/trunk/Source/VoxLib/IO/ResourceId.h)

The IO library provides basic mime type conversion functionality through an interface similar to the one used in Python. A singleton class MimeTypes manages a set of bindings between mime-types and file extensions controlled by the user.

# Option Sets #

**File:** [\_VoxLib/IO/OptionSet.h\_](https://code.google.com/p/voxrender/source/browse/trunk/Source/VoxLib/IO/ResourceId.h)

There are several IO library operations which utilize standard library multimaps to specify options. OptionSet is a wrapper for multimap which provides an interface for more easily managing and extracting option entries. It is recommended you make use of this class as future implementations of OptionSet may use boost::any if it can be shown to provide significant performance improvements.

# Adding IO Modules #

The VoxIO subset of VoxLib provides a standard API for module registration and usage as outlined in the documentation. For most purposes, the variety of protocols supported by the StandardIO plugin will be sufficient. These protocols include those provided by the LibCURL library:

  * HTTP(S):  [access remove query ](.md)
  * FTP(S):   [access remove query ](.md)
  * DICT:     [access remove query ](.md)
  * LDAP(S):  [access remove query ](.md)
  * IMAP(S):  [access remove query ](.md)
  * POP3(S):  [access remove query ](.md)
  * SMTP(S):  [access remove query ](.md)
  * FILE:     [access remove query ](.md)
  * GOPHER:   [access remove query ](.md)
  * TELNET:   [access remove query ](.md)
  * TFTP:     [access remove query ](.md)
  * SFTP:     [access remove query ](.md)
  * RTMP:     [access remove query ](.md)
  * RTSP:     [access remove query ](.md)
  * SCP:      [access remove query ](.md)

# Example Usage #
```
    // Register the resource opener modules
    vox::Resource::registerModule("file", FilesystemIO::create());

    // Establish the URI of the resource
    //
    // The IO library will automatically append a base URI
    // to any resource operation which is an application
    // default of the file protocol on the local filesystem.
    // Note that there is a distinction between the local filesystem,
    // no host, and the localhost authority in the identifier.
    ResourceId identifier = "file:///C:/my/path/file.xml";

    // Import the specified scene file
    try
    {
        ResourceOStream out(identifier);

        // Perform processing of scene //
    }
    catch(Error & error) { vox::Logger::addEntry(error); }
```