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

// Include Header
#include "ActionManager.h"

// Include Dependencies
#include "VoxLib/Core/Common.h"
#include "VoxLib/Error/Error.h"

namespace vox {

namespace {
namespace filescope {

    // A node in the action history which stores branch information
    class ActionNode
    {
    public:
        static std::shared_ptr<ActionNode> create(std::shared_ptr<Action> p_action)
        {
            std::shared_ptr<ActionNode> node(new ActionNode());
            node->action = p_action;
            return node;
        }

        std::list<std::shared_ptr<ActionNode>> branches; /// Oldest -> Newest :: Front -> Back

        std::shared_ptr<Action> action;
    };

    // Basic action class element
    class BasicAction : public Action
    {
    public: 
        // Creates a basic action element
        static std::shared_ptr<Action> create(
            std::function<void()> undo,
            std::function<void()> redo)
        {
            return std::make_shared<BasicAction>(undo, redo);
        }

        // Implementation of undo functionality
        void undo() { m_undoFunction(); }

        // Implementation of redo functionality
        void redo() { m_redoFunction(); }

        // Constructor for BasicAction class
        BasicAction(
            std::function<void()> undo,
            std::function<void()> redo) :
                Action(":TODO:"),
                m_undoFunction(undo),
                m_redoFunction(redo)
        {
        }

    private:
        std::function<void()> m_undoFunction;
        std::function<void()> m_redoFunction;
    };

} // namespace filescope
} // namespace anonymous

// ----------------------------------------------------------------------------
//  Implementation class containing action manager private members
// ----------------------------------------------------------------------------
class ActionManager::Impl
{
public:
    Impl() : maxBranch(1), maxDepth(50) { }

    std::list<std::shared_ptr<filescope::ActionNode>> undoStack;

    std::function<void()> callback;
    
    unsigned int maxDepth;  ///< Max depth of the undo/redo stacks
    unsigned int maxBranch; ///< Max branches of the action nodes
};

// ----------------------------------------------------------------------------
//  Initializes the implementation object
// ----------------------------------------------------------------------------
ActionManager::ActionManager() : m_pImpl(new Impl()) 
{ 
    m_pImpl->undoStack.push_back(filescope::ActionNode::create(nullptr));
}

// ----------------------------------------------------------------------------
//  Deletes the implementation object
// ----------------------------------------------------------------------------
ActionManager::~ActionManager() { delete m_pImpl; }

// ----------------------------------------------------------------------------
//  Undoes the action associated with the current node and moves up the tree
// ----------------------------------------------------------------------------
bool ActionManager::undo()
{
    if (!canUndo()) return false;

    bool stateChange = !canRedo();

    m_pImpl->undoStack.back()->action->undo();

    m_pImpl->undoStack.pop_back();

    if (!canUndo()) stateChange = true;

    if (stateChange) m_pImpl->callback();

    return true;
}

// ----------------------------------------------------------------------------
//  Redoes the action on the current node and moves back in the tree
// ----------------------------------------------------------------------------
bool ActionManager::redo(unsigned int branch)
{
    if (m_pImpl->undoStack.empty()) return false;

    auto & branches = m_pImpl->undoStack.back()->branches;

    if (branches.empty()) return false;

    bool stateChange = !canUndo();

    if (branches.size() < branch) throw Error(__FILE__, __LINE__, 
        VOX_LOG_CATEGORY, "Branch index out of range", Error_Range);

    auto iter = branches.begin();
    unsigned int index = branch;
    while (branch--) ++iter;

    m_pImpl->undoStack.push_back(*iter);

    m_pImpl->undoStack.back()->action->redo();

    if (m_pImpl->undoStack.size() > m_pImpl->maxDepth) 
    {
        m_pImpl->undoStack.pop_front();
        m_pImpl->undoStack.front()->action = nullptr;
    }

    if (canRedo()) stateChange = true;

    if (stateChange) m_pImpl->callback();

    return true;
}

// ----------------------------------------------------------------------------
//  Pushes a new action onto the action manager 
// ----------------------------------------------------------------------------
void ActionManager::push(std::shared_ptr<Action> action, bool doNow)
{
    auto node = filescope::ActionNode::create(action);
    
    if (doNow) action->redo();

    bool stateChange = !canUndo();

    auto & branches = m_pImpl->undoStack.back()->branches;
    branches.push_back(node);
    if (branches.size() > m_pImpl->maxBranch) 
        branches.pop_front();

    m_pImpl->undoStack.push_back(node);
    
    if (stateChange) m_pImpl->callback();
}

// ----------------------------------------------------------------------------
//  Pushes a new action onto the action manager
// ----------------------------------------------------------------------------
void ActionManager::push(std::function<void()> undo, std::function<void()> redo, bool doNow)
{
    push(filescope::BasicAction::create(undo, redo), doNow);
}

// ----------------------------------------------------------------------------
//  Clears the entire history from the action manager
// ----------------------------------------------------------------------------
void ActionManager::clear()
{
    bool stateChange;
    if (canUndo() || canRedo()) stateChange = true; 

    m_pImpl->undoStack.clear();
    
    m_pImpl->undoStack.push_back(filescope::ActionNode::create(nullptr));

    if (stateChange) m_pImpl->callback();
}

// ----------------------------------------------------------------------------
//  Returns the info ptr associated with the current node
// ----------------------------------------------------------------------------
String const& ActionManager::name()
{
    if (m_pImpl->undoStack.size() <= 1) throw Error(__FILE__, __LINE__, 
        VOX_LOG_CATEGORY, "No current node available", Error_NotAllowed);

    return m_pImpl->undoStack.back()->action->name();
}

// ----------------------------------------------------------------------------
//  Returns true if the undo stack is non-empty
// ----------------------------------------------------------------------------
bool ActionManager::canUndo() { return m_pImpl->undoStack.size() > 1; }

// ----------------------------------------------------------------------------
//  Returns true if the undo stack is non-empty
// ----------------------------------------------------------------------------
int ActionManager::canRedo() 
{ 
    return m_pImpl->undoStack.back()->branches.size(); 
}

// ----------------------------------------------------------------------------
//  Sets the maximum number of branches in a node
// ----------------------------------------------------------------------------
void ActionManager::setMaxBranches(unsigned int branches) { m_pImpl->maxBranch = branches; }

// ----------------------------------------------------------------------------
//  Sets the maximum depth of the tree
// ----------------------------------------------------------------------------
void ActionManager::setMaxDepth(unsigned int nodes) { m_pImpl->maxDepth = nodes; }

// ----------------------------------------------------------------------------
//  Returns the max number of branches
// ----------------------------------------------------------------------------
unsigned int ActionManager::maxBranches() { return m_pImpl->maxBranch; }

// ----------------------------------------------------------------------------
//  Returns the max depth of the tree
// ----------------------------------------------------------------------------
unsigned int ActionManager::maxDepth() { return m_pImpl->maxDepth; }

// ----------------------------------------------------------------------------
//  Sets the callback for action history state changes
// ----------------------------------------------------------------------------
void ActionManager::onStateChanged(std::function<void()> callback) { m_pImpl->callback = callback; }

} // namespace vox