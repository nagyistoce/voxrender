/* ===========================================================================

	Project: VoxScene

	Description: Defines the Scene class used by the Renderer

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
#include "Animator.h"

// Include Dependencies
#include "VoxLib/Scene/Scene.h"

namespace vox {

    class Animator::Impl
    {
    public:


        std::list<std::shared_ptr<KeyFrame>> m_keys;
    };

Animator::Animator() : m_pImpl(new Impl()) { }
Animator::~Animator() { delete m_pImpl; }

} // namespace vox