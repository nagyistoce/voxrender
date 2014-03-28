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
#ifndef CAM_EDIT_ACT_H
#define CAM_EDIT_ACT_H

// Include Dependencies
#include "VoxLib/Action/Action.h"
#include "VoxLib/Core/Functors.h"
#include "VoxScene/Camera.h"

/** Camera edit action */
class CamEditAct : public vox::Action
{
public:
    CamEditAct(std::shared_ptr<vox::Camera> camera) : 
        Action("Camera Edit")
    {
        m_camera = vox::Camera::create();
        camera->clone(*m_camera);
    }

    virtual void undo();

    virtual void redo() { undo(); }

private:
    std::shared_ptr<vox::Camera> m_camera;
};

// End Definition
#endif // CAM_EDIT_ACT_H
