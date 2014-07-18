/* ===========================================================================

	Project: VoxRender

	Description: Implements an action for undo/redo operations

    Copyright (C) 2014 Lucas Sherman

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

// Begin Definition
#ifndef ADD_REM_KEY_ACT_H
#define ADD_REM_KEY_ACT_H

// Include Dependencies
#include "VoxLib/Action/Action.h"
#include "VoxLib/Core/Functors.h"
#include "VoxScene/Animator.h"
#include "VoxScene/Scene.h"

/** Camera edit action */
class AddRemKeyAct : public vox::Action
{
public:
    AddRemKeyAct(int index, std::shared_ptr<vox::KeyFrame> keyframe, bool add) : 
        m_keyframe(keyframe),
        m_isAdd(add),
        m_index(index)
    {
    }

    static std::shared_ptr<AddRemKeyAct> create(int index, std::shared_ptr<vox::KeyFrame> keyframe, bool add)
    {
        return std::make_shared<AddRemKeyAct>(index, keyframe, add);
    }

    virtual void undo() { if (m_isAdd) rem(); else add(); }

    virtual void redo() { if (m_isAdd) add(); else rem(); }

private:
    void add();
    void rem();

private:
    std::shared_ptr<vox::KeyFrame> m_keyframe;
    bool m_isAdd;
    int m_index;
};

// End Definition
#endif // ADD_REM_KEY_ACT_H
