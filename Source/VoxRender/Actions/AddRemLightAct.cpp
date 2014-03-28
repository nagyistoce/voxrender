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

// Include Header
#include "AddRemLightAct.h"

// Include Dependencies
#include "VoxLib/Error/Error.h"
#include "Interface/mainwindow.h"

// ----------------------------------------------------------------------------
//  Reverses the effects of a camera edit action
// ----------------------------------------------------------------------------
void AddRemLightAct::undo()
{
    auto scene = MainWindow::instance->scene();
    if (!scene.lightSet)
    {
        throw vox::Error(__FILE__, __LINE__, "GUI", "Current scene light set is missing", vox::Error_Bug);
        return;
    }

    vox::SceneLock lock(scene.lightSet);

    if (scene.lightSet->find(m_light->id()))
    {
        scene.lightSet->remove(m_light, true);
        
        scene.lightSet->setDirty();
    }
    else 
    {
        scene.lightSet->add(m_light, true);

        m_light->setDirty();
    }
}