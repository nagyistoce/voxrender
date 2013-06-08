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

namespace vox {

namespace {
namespace filescope {

    // :TODO: Move to IMPL header
    /** Basic action class element */
    class BasicAction : Action
    {
    public: 
        /** Creates a basic action element */
        static std::shared_ptr<BasicAction> makeAction(
            std::function<void()> undo,
            std::function<void()> redo,
            bool doNow)
        {
            return std::make_shared<BasicAction>(undo, redo, doNow);
        }

        /** Implementation of undo functionality */
        void undo() { m_undoFunction(); }

        /** Implementation of redo functionality */
        void redo() { m_redoFunction(); }

        /** Constructor for BasicAction class */
        BasicAction(
            std::function<void()> undo,
            std::function<void()> redo,
            bool doNow) :
                m_undoFunction(undo),
                m_redoFunction(redo)
        {
             if (doNow) m_redoFunction();
        }

    private:
        std::function<void()> m_undoFunction;
        std::function<void()> m_redoFunction;
    };

    // :TODO: Move to IMPL header
    /** A node in the action history which stores branch information */
    class ActionNode
    {
        std::list<std::shared_ptr<Action>> m_branches; /// Oldest -> Newest == Front -> Back
    };

    // Stack for action history information (Actually a list for
    // dropping history)
    static std::list<ActionNode> m_actionHistory;

} // namespace filescope
} // namespace anonymous

} // namespace vox