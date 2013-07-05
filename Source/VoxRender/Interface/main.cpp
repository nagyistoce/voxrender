/* ===========================================================================

	Project: VoxRender - Interactive GPU Based Volume Rendering

	Description: Performs interactive rendering of volume data using 
		photon mapping and volume ray casting techniques.

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

	The VoxRender GUI is derived from LuxRender's GUI (implemented with QT)
	Lux Renderer website : http://www.luxrender.net
    QT Website : http://qt.digia.com/

=========================================================================== */

// Include Application
#include "voxapp.h"

// Application Entry Point
int main(int argc, char** argv) 
{
    VoxGuiApp application(argc, argv);

	application.initialize();

	// Execute application
    return application.exec();
}