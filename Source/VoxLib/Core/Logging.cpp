/* ===========================================================================

	Project: VoxRender - Logging interface

	Description:
	 Implements an front-end for run-time logging and error handling. 
     The log backend prints to std::clog by default, but is modifiable via
     a static member of the Logger class.

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
#include "Logging.h"

// Include dependencies
#include "VoxLib/Core/Common.h"
#include "VoxLib/Core/System.h"

// Disable MSVC warnings
#ifdef _MSC_VER
#pragma warning (disable: 4996) // Deprecated function warnings
#endif // _WIN32

// Win32 Console Color Utility
#ifdef _WIN32
#   define NOMINMAX
#   include <windows.h>
#   include <stdio.h>
#   include <wincon.h>
#   define FOREGROUND_YELLOW (FOREGROUND_RED | FOREGROUND_GREEN)
#   define FOREGROUND_WHITE (FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE)
#endif

// Filescope namespace
namespace {
namespace filescope {
        
        // -----------------------------------------------------
        //  Sets the current console color
        // -----------------------------------------------------
#ifdef _WIN32
    inline void changeConsoleColor( WORD col )
	{
		HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
		CONSOLE_SCREEN_BUFFER_INFO screenBufferInfo;
		GetConsoleScreenBufferInfo( hConsole, &screenBufferInfo );
		col |= screenBufferInfo.wAttributes & static_cast<WORD>(FOREGROUND_INTENSITY | BACKGROUND_INTENSITY);
		SetConsoleTextAttribute( hConsole, col );
	}
#endif

} // namespace filescope
} // namespace anonymous

// API namespace
namespace vox {

//  Static member initialization
ErrorHandler Logger::m_errorHandler = ErrorPrint;
int Logger::m_filter                = Severity_Info;
int Logger::m_lastError             = Error_None;

// -----------------------------------------------------
//  Dispatches a log entry to the logger backend
// -----------------------------------------------------
LogEntry::~LogEntry()
{ 
    Logger::addEntry(m_severity, m_code, m_category, 
                     str().c_str(), m_file, m_line); 
}

// -----------------------------------------------------
//  Ignores any errors regardless of severity
// -----------------------------------------------------
void ErrorIgnore(char const* file, int line, int severity, int code,
                 char const* category, char const* message)
{
}

// -----------------------------------------------------
//  Aborts the program if an error occurs
// -----------------------------------------------------
void ErrorAbort(char const* file, int line, int severity, int code,
                char const* category, char const* message)
{
	ErrorPrint(file, line, severity, code, category, message);
	if (severity >= Severity_Error) exit(code);
}

// -----------------------------------------------------
//  Outputs a log message to the std::clog stream
// -----------------------------------------------------
void ErrorPrint(char const* file, int line, int severity, int code,
                     char const* category, char const* message)
{
	std::clog << "[";

#ifdef _WIN32
	switch (severity) 
	{
    case Severity_Trace:
        filescope::changeConsoleColor( FOREGROUND_WHITE );
        break;
	case Severity_Info:
		filescope::changeConsoleColor( FOREGROUND_GREEN );
		break;
	case Severity_Warning:
		filescope::changeConsoleColor( FOREGROUND_YELLOW );
		break;
	case Severity_Error:
		filescope::changeConsoleColor( FOREGROUND_RED );
		break;
	case Severity_Fatal:
		filescope::changeConsoleColor( FOREGROUND_RED );
		break;
	case Severity_Debug:
		filescope::changeConsoleColor( FOREGROUND_BLUE );
		break;
    default:
        // Unknown severity level //
        break;
	}
#endif

    std::clog << boost::local_time::local_sec_clock::local_time(
        boost::local_time::time_zone_ptr( ) ) << " ";

	switch (severity) 
	{
	case Severity_Debug:
		std::clog << "DEBUG";
		break;
	case Severity_Info:
		std::clog << "INFO";
		break;
	case Severity_Warning:
		std::clog << "WARNING";
		break;
	case Severity_Error:
		std::clog << "ERROR";
		break;
	case Severity_Fatal:
		std::clog << "FATAL";
		break;
	}
	std::clog << " " << code;

    // Restore default previous console color :TODO: restore previous
#ifdef _WIN32
	filescope::changeConsoleColor( FOREGROUND_WHITE );
#endif

	std::clog << "] "; 
    
    std::clog << System::getCurrentPid() << " "; 
    
    std::clog << System::getCurrentTid() << " ";

    std::clog << category << " ";
    
    std::clog << message; 
   
    std::clog << " <file=\"" << boost::filesystem3::basename(file) << "\",line=" << line << ">";
        
    std::clog << std::endl << std::flush;
}

} // namespace vox