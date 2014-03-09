/* ===========================================================================

	Project: VoxRender

	Description: Generic dialogue for option set specification by user

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
#ifndef GENERIC_DIALOGUE_H
#define GENERIC_DIALOGUE_H

// Include Qt Dependencies
#include <QDialog>

// Include Dependencies
#include "VoxVolt/Filter.h"

namespace Ui {
class GenericDialogue;
}

/** Generic dialogue for filter parameter specification */
class GenericDialogue : public QDialog
{
    Q_OBJECT
    
public:
    GenericDialogue(vox::String const& title, std::list<vox::volt::FilterParam> params);

    void getOptions(vox::OptionSet & options);

    ~GenericDialogue();

private:
    Ui::GenericDialogue *ui;

    std::list<vox::volt::FilterParam> m_parameters;
};

#endif // GENERIC_DIALOGUE_H
