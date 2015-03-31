# Contents #



# Overview #

Rendering with VoxLib is performed by inserting an implementation of a !Renderer interface into a RenderController. A RenderController then maintains a persistant thread of execution in which it manages and executes a collection of Renderers.

:TODO: Diagram Here

# Setting up a RenderController #

**File:** [VoxLib/Rendering/RenderController.h](https://code.google.com/p/voxrender/source/browse/trunk/Source/VoxLib/Rendering/RenderController.h)

The first step in setting up a rendering environment is to create a RenderController which will manage your collection of Renderers.
A RenderController maintains a set of handles (std::shared\_ptr) to Renderer objects which it is authorized to utilize. One of these handles is designated a master and the rest slaves. These designations are described in detail in the [Renderers#Renderers](Renderers#Renderers.md) section below.

  * To add or remove slaves from your controller, you can use the member functions _RenderController::addRenderer_ and _RenderController::removeRenderer_. These functions are thread safe and permit the modification of the authorized member set for a Controller during rendering.

  * The master renderer is specified at the initiation of rendering operations using the member function _RenderController::render_. It can only be removed from the controller by internal failure detection or termination of the render using _RenderController::stop_ and _RenderController::reset_ to reset the state of the controller.



# Renderers #