/* ===========================================================================

	Project: VoxRender - Application Interface

	Implements the primary application window and GUI error handler
	Based on luxrender main window interface classes.
	Lux Renderer website : http://www.luxrender.net 

	Description:
	 Implements an interface for run-time logging and error handling

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
#ifndef VOX_APP_H
#define VOX_APP_H

// Include Dependencies
#include "mainwindow.h"

// QT4 Includes
#include <QtGui/QApplication>

// VoxRender Application Class
class VoxGuiApp : public QApplication
{
	Q_OBJECT

public:
	VoxGuiApp( int argc, char **argv );
	~VoxGuiApp( );

	MainWindow *mainwindow;
	void initialize( );

private:
	int m_argc; char **m_argv;
};

#endif // VOX_APP_H
