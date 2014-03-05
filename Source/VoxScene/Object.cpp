/* ===========================================================================

	Project: VoxScene

	Description: Defines a basic identifiable scene element

    Copyright (C) 2012-2014 Lucas Sherman

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
#include "Object.h"

// Include Dependencies
#include "boost/atomic/atomic.hpp"
#include "boost/thread.hpp"

namespace vox {

namespace {
namespace filescope {

    boost::atomic<int> uidCounter(1);

} // namespace filescope
} // namespace anonymous

Object::Object() : m_mutex(new boost::mutex()) { m_id = filescope::uidCounter++; }

Object::Object(int id) : m_mutex(new boost::mutex()) { m_id = id; }

Object::~Object() { delete m_mutex; }

void Object::lock() { m_mutex->lock(); }
        
void Object::unlock() { m_mutex->unlock(); }

} // namespace vox
