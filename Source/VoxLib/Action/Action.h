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
#ifndef VOX_ACTION_H
#define VOX_ACTION_H

// Include Dependencies
#include "VoxLib/Core/CudaCommon.h"

/** Action class used by ActionManager */
class VOX_EXPORT Action
{
public:
    /** Virtualized destructor for inheritance */
    virtual ~Action() {}

    /** Reverse the effects of this action */
    virtual void undo() = 0;

    /** Reapplies the effects of this action */
    virtual void redo() = 0;

    /** Returns the user info */
    std::shared_ptr<void*> userInfo()
    {
        return m_userInfo;
    }

protected:
    std::shared_ptr<void*> m_userInfo;
};

// End definition
#endif // VOX_ACTION_H