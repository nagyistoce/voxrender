/* ===========================================================================

	Project: VoxRender 

	Description: Interactive Volume Rendering

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

// Include Headers
#include "clipdialogue.h"
#include "ui_clipdialogue.h"

// --------------------------------------------------------------------
//  Initialize the light dialogue check state
// --------------------------------------------------------------------
ClipDialogue::ClipDialogue(int index) :
    QDialog(nullptr),
    ui(new Ui::ClipDialogue)
{
    ui->setupUi(this);

    // Append the light number to the name
    ui->lineEdit_name->setText(QString("Clip_%1").arg(index));

    // Select the default (Point) light
    ui->radioButton_plane->setChecked(true);
}
    
// --------------------------------------------------------------------
//  Initialize the light dialogue check state
// --------------------------------------------------------------------
ClipDialogue::~ClipDialogue()
{
    delete ui;
}

// --------------------------------------------------------------------
//  Returns the name selected for the light
// --------------------------------------------------------------------
QString ClipDialogue::nameSelected()
{
    return ui->lineEdit_name->text();
}

// --------------------------------------------------------------------
//  Returns the type of button selected in the dialogue
// --------------------------------------------------------------------
ClipType ClipDialogue::typeSelected()
{
    if (ui->radioButton_plane->isChecked())
    {
        return ClipType_Plane;
    }
    else
    {
        return ClipType_Sphere;
    }
}