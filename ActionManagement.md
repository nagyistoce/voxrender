# Contents #



# Action Manager #

**File:** https://code.google.com/p/voxrender/source/browse/trunk/Source/VoxLib/Action/ActionManager.h

The ActionManager is intended to be an interface with which to implement undo/redo functionality within an application. Implementations of Action objects are pushed onto the ActionManager and stored in a tree through for use with the undo, redo, and branch operations.

# Actions #

**File:** https://code.google.com/p/voxrender/source/browse/trunk/Source/VoxLib/Action/Action.h

An Action object is a virtual base from which implementations of undo/redo-able actions within a program can inherit. Action objects are added to the action history of an application through the member function ActionManager::push(). Note that it is also possible to directly register undo/redo functions with the manager using an overload of ActionManager::push().

Action objects are removed from the manager as they exceed the maximum memory constraints imposed by the user.

# Resource Management #

Both push overloads of the ActionManager utilize shared memory patterns. As new actions are pushed to the manager, shared\_ptrs and functions associated will old actions will be released. If memory is a concern, than it is up to the user to select an appropriate configuration and apply it using the relevant methods:

ActionManager::setMaxBranches and ActionManager::setMaxDepth

# Example Usage #
```

```