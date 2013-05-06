/* ===========================================================================

	Project: VoxRender - QT Interface Utility Functions

	Description: Implements some utilities for QT control management

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

// Include Header
#include "utilities.h"

// QT Includes
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QSlider>

// --------------------------------------------------------------------
//  Updates a slider control to reflect the doubleSpinBox control's value
// --------------------------------------------------------------------
void Utilities::forceSbToSl(QDoubleSpinBox * spinBox, QSlider * slider, int value)
{
    auto smax = slider->maximum();
    auto smin = slider->minimum();
    auto bmax = spinBox->maximum();
    auto bmin = spinBox->minimum();

    auto transfer = (smax - smin) / (bmax - bmin);

    double bvalue = static_cast<double>(bmin + (value - smin) * transfer);

    bool old = spinBox->blockSignals(true);
    spinBox->setValue(bvalue);
    spinBox->blockSignals(old);
}

// --------------------------------------------------------------------
//  Updates a doubleSpinBox control to reflect the slider control's value
// --------------------------------------------------------------------
void Utilities::forceSlToSb(QSlider * slider, QDoubleSpinBox * spinBox, double value)
{
    auto smax = slider->maximum();
    auto smin = slider->minimum();
    auto bmax = spinBox->maximum();
    auto bmin = spinBox->minimum();
    
    auto transfer = (bmax - bmin) / (smax - smin);

    int svalue = static_cast<int>(smin + (value - bmin) * transfer);

    bool old = slider->blockSignals(true);
    slider->setValue(svalue);
    slider->blockSignals(old);
}
