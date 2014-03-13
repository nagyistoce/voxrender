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
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QSlider>

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
    /** Constructs a dialog window displaying the specified filter options */
    GenericDialogue(vox::String const& title, std::list<vox::volt::FilterParam> params);

    /** Returns an option set of the user defined filter parameters */
    void getOptions(vox::OptionSet & options);

    /** Destructor */
    ~GenericDialogue();

private:
    Ui::GenericDialogue *ui;

    std::list<vox::volt::FilterParam> m_parameters; ///< The filter parameters to display for the user

    std::map<QDoubleSpinBox*,QSlider*> m_connMap; ///< Mappings between double/int, spinBox/sliders

private slots:
    /** Redirection slot for int->double slider spinBox connections */
    void valueChangeRedirect(int value);
    
    /** Redirection slot for int->double slider spinBox connections */
    void valueChangeRedirect(double value);
};

#endif // GENERIC_DIALOGUE_H
