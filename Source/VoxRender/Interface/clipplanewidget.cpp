/* ===========================================================================

	Project: VoxRender

	Description: Implements the interface for point light source settings

    Copyright (C) 2012-2013 Lucas Sherman

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

// Include Headers
#include "ui_clipplanewidget.h"
#include "clipplanewidget.h"

// Include Dependencies
#include "utilities.h"

// VoxLib Dependencies
#include "VoxLib/Core/Geometry/Primitives.h"
#include "VoxLib/Core/format.h"

// QT Includes
#include <QtWidgets/QMessageBox>

using namespace vox;

// File scope namespace
namespace {
namespace filescope {
}
}

// --------------------------------------------------------------------
//  Constructor - Initialize the widget ui
// --------------------------------------------------------------------
ClipPlaneWidget::ClipPlaneWidget(QWidget * parent, std::shared_ptr<Plane> plane) : 
    QWidget(parent), 
    ui(new Ui::ClipPlaneWidget),
    m_title("Clipping Plane"),
    m_plane(plane)
{
	ui->setupUi(this);

    m_dirty = false;
}

// --------------------------------------------------------------------
//  Destructor
// --------------------------------------------------------------------
ClipPlaneWidget::~ClipPlaneWidget()
{
    delete ui;
}

// --------------------------------------------------------------------
//  Synchronizes the light widget's position controls with the scene
// --------------------------------------------------------------------
void ClipPlaneWidget::processInteractions()
{
    if (m_dirty)
    {
        m_dirty = false; // Reset tracking before read sequence //
    }
}