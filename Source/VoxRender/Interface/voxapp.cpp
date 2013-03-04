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

// Include Header 
#include "voxapp.h"

// Constructor - Stores the commandline arguments for
// parsing on initialization
VoxGuiApp::VoxGuiApp( int argc, char **argv ) : 
	m_argc(argc), m_argv(argv), QApplication( argc, argv ) { }

// Destructor - Delete the main window
VoxGuiApp::~VoxGuiApp( )
{
	if( mainwindow != NULL )
		delete mainwindow;
}

// Initialize - Initializes the main window and parses
// the commandline options
void VoxGuiApp::initialize( )
{
	// Initialize the main window
	mainwindow = new MainWindow( );
	mainwindow->show( );
}