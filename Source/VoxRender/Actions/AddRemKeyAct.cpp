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
#include "AddRemKeyAct.h"

// Include Dependencies
#include "VoxLib/Error/Error.h"
#include "Interface/mainwindow.h"

// ----------------------------------------------------------------------------
//  Reverses the effects of an add action
// ----------------------------------------------------------------------------
void AddRemKeyAct::add()
{
    auto scene = MainWindow::instance->scene();
    if (!scene.animator)
    {
        throw vox::Error(__FILE__, __LINE__, "GUI", "Current scene animator is missing", vox::Error_Bug);
        return;
    }

    scene.animator->addKeyframe(m_keyframe, m_index, true);
}

// ----------------------------------------------------------------------------
//  Reverses the effects of a remove action
// ----------------------------------------------------------------------------
void AddRemKeyAct::rem()
{
    auto scene = MainWindow::instance->scene();
    if (!scene.animator)
    {
        throw vox::Error(__FILE__, __LINE__, "GUI", "Current scene animator is missing", vox::Error_Bug);
        return;
    }

    scene.animator->removeKeyframe(m_index, true);
}