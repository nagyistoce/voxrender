/* ===========================================================================

	Project: VoxLib - Action Manager
    
	Description: Acts as a global action management tree for redo/undo/branch

    Copyright (C) 2013 Lucas Sherman

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

// Begin definition
#ifndef VOX_ACTION_MANAGER_H
#define VOX_ACTION_MANAGER_H

// Include Dependencies
#include "VoxLib/Core/CudaCommon.h"
#include "VoxLib/Action/Action.h"

namespace vox {

/** 
 * A class for managing an undo/redo/branch history of an application 
 *
 * This class stores an internal tree of all registered actions for the purposes of initiating
 * undo, redo, or push operations at the current point in the tree.
 *
 * - A push operation adds a new action to the action history and sets the current node to
 *   this action
 *
 * - An undo operation calls the undo member of the action at the current node and sets the
 *   new current node to the previous node
 *
 * - A redo operation call the redo member of the action at the current node and sets the
 *   current node to a more recent node in the tree which was pushed at this location. The 
 *   new current node is defined by the branch parameter of the redo member function.
 *
 * If executing an operation would exceed the history depth or breadth limitations of the history, 
 * older actions will first be removed from the history.
 */
class VOX_EXPORT ActionManager
{
public:
    // :TODO: State Change Callbacks

    /** 
     * Adds a new action to the action history 
     *
     * Registers a new event with the Action Manager. If this event exceeds the depth or breadth 
     * limitations imposed on the tree, the oldest nodes will be removed. While the insertion of 
     * action elements is thread safe, the order of insertion is, as a result, unspecified.
     */
    static void push(std::shared_ptr<Action> action);
    
    /** 
     * Function based overload for push 
     *
     * @param undo  The undo function for the action
     * @param redo  The redo function for the action
     * @param doNow If true, the redo function will be called immediately
     * 
     */
    static void push(std::function<void()> undo, std::function<void()> redo, bool doNow);
    
    /** 
     * Reverse the effects of the last action performed, if any 
     *
     * @returns true if an action was undone, false otherwise
     */
    static bool undo();

    /** 
     * Redoes the effects of the last action undone, if any 
     * :TODO: Define the exception handling internally, ie let the user drop a failed action
     *        and retain the stack or clear the stack entirely, etc... ?s
     *
     * @param branch The branch to redo
     *
     * @returns true if an action was undone, false otherwise
     */
    static bool redo(unsigned int branch = 0);
    
    /** Returns true if there is a valid current node in the history */
    static bool canUndo();

    /** Returns true if there is a valid branch at the current node */
    static bool canRedo();

    /** Returns the content of the userInfo element at the current action */
    static std::shared_ptr<void*> info();

    /** Clears the action manager history */
    static void clear();

    /** Sets the maximum number of additional branches per node (Not counting the original branch) */
    static void setMaxBranches(unsigned int branches);

    /** Sets the maximum depth of the action history stack */
    static void setMaxDepth(unsigned int nodes);

    /** Returns the current maximum branches per node (Not counting the original branch) */
    static unsigned int maxBranches();

    /** Returns the current maximum stack depth */
    static unsigned int maxDepth();
};

} // namespace vox

// End definition
#endif // VOX_ACTION_MANAGER_H