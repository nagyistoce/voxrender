/* ===========================================================================

	Project: Standard Volume Filters
    
	Description: Exposes some standard filters provided by the volt library

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

// Begin definition
#ifndef SVF_VIEWS_H
#define SVF_VIEWS_H

// Include Dependencies
#include "Plugins/StdVolumeFilter/Common.h"
#include "VoxLib/Core/CudaCommon.h"
#include "VoxLib/Error/Error.h"
#include "VoxVolt/Core.h"

// API namespace
namespace vox
{

/** Laplacian filter provided by Volt library */
class View : public volt::Filter
{
public:
    enum Dir
    {
        Dir_Begin,
        Dir_Front = Dir_Begin,
        Dir_Left,
        Dir_Top,
        Dir_Back,
        Dir_Right,
        Dir_Bottom,
        Dir_End,
    };

    static const char* dirStrings[];

public:
    View(std::shared_ptr<void> handle, int dir) : m_handle(handle), m_dir(dir) { }

    String name() { return String("Camera.") + dirStrings[m_dir]; }

    void execute(Scene & scene, OptionSet const& params);

private:
    std::shared_ptr<void> m_handle;

    int m_dir;
};

} // namespace vox

// End definition
#endif // SVF_VIEWS_H