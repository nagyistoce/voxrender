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

// Begin definition
#ifndef LIGHTDIALOGUE_H
#define LIGHTDIALOGUE_H

// Include Qt Dependencies
#include <QDialog>

namespace Ui {
class LightDialogue;
}

enum LightType
{
    LightType_Error       = 0,
    LightType_Point       = 1,
    LightType_Area        = 2,
    LightType_Environment = 3,
    LightType_Spot        = 4
};

class LightDialogue : public QDialog
{
    Q_OBJECT
    
public:
    explicit LightDialogue(int index);
    ~LightDialogue();
   
    LightType typeSelected( );
    QString nameSelected( ); 

private:
    Ui::LightDialogue *ui;

private slots:
    void toggleEnv(bool checked);
    void toggleArea(bool checked);
    void togglePoint(bool checked);
    void toggleSpot(bool checked);
};

#endif // LIGHTDIALOGUE_H
