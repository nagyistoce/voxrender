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

// Begin Definition
#ifndef UTILITIES_H
#define UTILITIES_H

class QDoubleSpinBox;
class QSlider;

/** QT Utility Functions */
class Utilities
{
public:
    /** Forces the relative value of a spinbox to match that of a slider */
    static void forceSbToSl(QDoubleSpinBox * spinBox, QSlider * slider, int value);

    /** Forces the relative value of a slider to match that of a spinbox */
    static void forceSlToSb(QSlider * slider, QDoubleSpinBox * spinBox, double value);
};

// End Definition
#endif // UTILITIES_H
