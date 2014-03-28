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
#ifndef MATERIAL_EDIT_ACT_H
#define MATERIAL_EDIT_ACT_H

// Include Dependencies
#include "VoxLib/Action/Action.h"
#include "VoxLib/Core/Functors.h"
#include "VoxScene/Material.h"

/** Camera edit action */
class MaterialEditAct : public vox::Action
{
public:
    MaterialEditAct(std::shared_ptr<vox::Material> material, 
                    std::shared_ptr<vox::Material> reference) : 
        Action("Material Edit"),
        m_material(material),
        m_reference(reference)
    {
    }

    virtual void undo();

    virtual void redo() { undo(); }

private:
    std::shared_ptr<vox::Material> m_material;
    std::shared_ptr<vox::Material> m_reference;
};

// End Definition
#endif // MATERIAL_EDIT_ACT_H
