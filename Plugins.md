# Contents #



# Unmanaged Plugins #

**File:** [VoxLib/Plugin/Plugin.h](https://code.google.com/p/voxrender/source/browse/trunk/Source/VoxLib/Plugin/Plugin.h)

## Library Management ##

The Plugin class essentially provides a platform independent mechanism for loading and managing shared libraries. Unlike plugins used by the PluginManager, there are no restrictions on the interface or operation of a loaded library.

The class interface provides wrappers for 3 basic operations on supported platforms:

  * Loading of library files
  * Unloading of library files
  * Symbol lookup within a loaded library

Load and unload functionality is managed by the member functions _Plugin::open_ and _Plugin::close_. There are also constructor overloads to call open automatically as well as a destructor which ensures library closure is accomplished. Symbol lookup can be performed by the lookup members of the Plugin class. They provide built-in support for typecasting and detection of missing symbols.

## Copying ##

The Plugin class provides both a copy constructor and a move constructor. The standard copy constructor will automatically increment the reference count on a shared library by making a call to the underlying OS specific load function. Depending on the target platform, users may find that they can improve performance by using the move constructor as necessary to avoid this overhead.

## Reloading ##

The member function _Plugin::reload_ allows a user to close and reopen a library in a single call without having to recapture the libraries filesystem location. It is subject to some conditions and repercussions.

  * If there are multiple handles to the underlying library, this call will have no effect.

  * If this call succeeds, the handles returned by any symbol lookups in the library will be invalidated.

# Plugin Management #

**File:** [VoxLib/Plugin/PluginManager.h](https://code.google.com/p/voxrender/source/browse/trunk/Source/VoxLib/Plugin/PluginManager.h)

## Interface Requirements ##

The plugin management interface provides a stricter interface for loading and running well defined plugins for an application. Plugins take the form of a shared library which provides the the following c-linkage interface:

```
 void initPlugin()             <<< Called immediately after plugin loading
 void freePlugin()             <<< Called before the plugin is unloaded
  
 void enable()                 <<< Called on user permission to activate
 void disable()                <<< Called on user shutdown request

 char const* name()            <<< Returns the name of the plugin
 char const* vendor()          <<< Returns the plugins creating organization
 char const* apiVersionMin()   <<< Returns the minimum supported API version
 char const* apiVersionMax()   <<< Returns the maximum supported API version
 char const* version()         <<< Returns the version of the plugin
```

The name component is used to distinguish plugins from each other. Plugins with the same name, but different version string will be evaluated to determine which plugin provides the most recent version but retains compatibility with the current API version.

# Example Usage #
```
    // Acquire the handle to the manager global instance
    auto pluginManager = vox::PluginManager::instance();

    // Initialize the plugin search directories
	// :NOTE: Relative paths will be applied to the
	//        current director as a base reference
	pluginManager.addDirectory("Plugins");
	
	// Load any plugins in the search directories
	// (If a potential plugin fails verification, it
	//  will automatically be unloaded before returning)
	pluginManager.loadAll();
	
	// Generate a list of the available plugins
    auto plugins = pluginManager.getInfo();
	
	// Run our program, perform some processing, etc
	doStuff();
	
	// Its often usefull to ensure all plugins are unloaded 
	// explicitly to prevent any issues with application 
	// pull down while plugins are running (Logging, etc)
	pluginManager.unloadAll();
```