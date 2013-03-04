/* ===========================================================================

	Project: VoxRender - Light Dialogue

	Description: Dialogue for light type selection (with description).

    Copyright (C) 2012 Lucas Sherman

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
#include "lightdialogue.h"
#include "ui_lightdialogue.h"

// --------------------------------------------------------------------
//  Initialize the light dialogue check state
// --------------------------------------------------------------------
LightDialogue::LightDialogue(int index) :
    QDialog(nullptr),
    ui(new Ui::LightDialogue)
{
    ui->setupUi(this);

    // Connect radio button toggles to description box update functions
    connect( ui->radioButton_area,        SIGNAL(toggled(bool)), this, SLOT(toggleArea(bool))  );
    connect( ui->radioButton_environment, SIGNAL(toggled(bool)), this, SLOT(toggleEnv(bool))   );
    connect( ui->radioButton_point,       SIGNAL(toggled(bool)), this, SLOT(togglePoint(bool)) );
    connect( ui->radioButton_spot,        SIGNAL(toggled(bool)), this, SLOT(toggleSpot(bool))  );

    // Append the light number to the name
    ui->lineEdit_name->setText( QString("Light_%1").arg(index) );

    // Select the default (Point) light
    ui->radioButton_point->setChecked(true);
}
    
// --------------------------------------------------------------------
//  Initialize the light dialogue check state
// --------------------------------------------------------------------
LightDialogue::~LightDialogue()
{
    delete ui;
}

// --------------------------------------------------------------------
//  Returns the name selected for the light
// --------------------------------------------------------------------
QString LightDialogue::nameSelected( )
{
    return ui->lineEdit_name->text( );
}

// --------------------------------------------------------------------
//  Returns the type of button selected in the dialogue
// --------------------------------------------------------------------
LightType LightDialogue::typeSelected( )
{
    if( ui->radioButton_point->isChecked( ) )
    {
        return LightType_Point;
    }
    else if( ui->radioButton_area->isChecked( ) )
    {
        return LightType_Area;
    }
    else if( ui->radioButton_spot->isChecked( ) ) 
    {
        return LightType_Spot;
    }
    else if( ui->radioButton_environment->isChecked( ) ) 
    {
        return LightType_Environment;
    }
    else
    {
        return LightType_Error;
    }
}

// --------------------------------------------------------------------
//  Updates the description text in the info field of the dialogue
// --------------------------------------------------------------------
void LightDialogue::toggleEnv(bool checked) 
{ 
    if (!checked) return;

    ui->textEdit_description->setText(
        "Environment Light"
        );
}

// --------------------------------------------------------------------
//  Updates the description text in the info field of the dialogue
// --------------------------------------------------------------------
void LightDialogue::toggleArea(bool checked) 
{ 
    if (!checked) return;

    ui->textEdit_description->setText(
        "Area Light"
        );
}

// --------------------------------------------------------------------
//  Updates the description text in the info field of the dialogue
// --------------------------------------------------------------------
void LightDialogue::togglePoint(bool checked) 
{ 
    if (!checked) return;

    ui->textEdit_description->setText(
        "Point Light"
        );
}

// --------------------------------------------------------------------
//  Updates the description text in the info field of the dialogue
// --------------------------------------------------------------------
void LightDialogue::toggleSpot(bool checked) 
{ 
    if (!checked) return;

    ui->textEdit_description->setText(
        "Spot Light"
        );
}