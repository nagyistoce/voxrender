/* ===========================================================================

	Project: VoxRender - Systems header

	Description: Provides access to OS or system specific information

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
#include "System.h"

// Additional dependencies
#include <boost/filesystem.hpp>

// Include OS specific required headers
#if defined _WIN32
#   define NOMINMAX
#   include <windows.h>
#   include <LMCons.h>
#elif defined __linux__
#   include <unistd.h>
#   include <sys/types.h>
#endif

// API namespace
namespace vox
{
    
// --------------------------------------------------------------------
//  Returns the current computer name if available
// --------------------------------------------------------------------
String System::computerName()
{
#ifdef _WIN32
    static const DWORD maxlen = MAX_COMPUTERNAME_LENGTH+1;

    Char buf[maxlen]; DWORD bufsize = maxlen;
    if (GetComputerName(buf, &bufsize))
    {
        return String(buf);
    }
#elif defined __linux__
    static const DWORD maxlen = HOST_NAME_MAX+1;
    Char buf[maxlen]; DWORD bufsize = maxlen;
    if(gethostname() = 0)
    {
        return String(buf);
    }
#else
#   pragma message "WARNING computerName not implemented for this system"
#endif

    return String();
}

// --------------------------------------------------------------------
//  Returns the current user name if available
// --------------------------------------------------------------------
String System::userName()
{
#ifdef _WIN32
    static const DWORD maxlen = UNLEN+1;

    Char buf[maxlen]; DWORD bufsize = maxlen;
    if (GetUserName(buf, &bufsize))
    {
        return String(buf);
    }
#else
#   pragma message "WARNING computerName not implemented for this system"
#endif

    return 0;
}
        
// --------------------------------------------------------------------
//  Returns the current directory
// --------------------------------------------------------------------
String System::currentDirectory()
{
    return boost::filesystem::current_path().string();
}

// --------------------------------------------------------------------
//  Returns the getLastError value as a info string 
// --------------------------------------------------------------------
String System::formatError(size_t error)
{
#if defined _WIN32
    Char* buf = nullptr;
    DWORD bufsize = FormatMessage( 
         FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
         NULL, error, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
         buf, NULL, NULL
         );

    if (bufsize)
    {
        String result(buf, bufsize);
        LocalFree(buf);
        return result;
    }
#elif defined __linux__
#   return String(get_current_dir_name());
#else
#   pragma message "WARNING formatError not implemented for this system"
#endif
    
    return std::string();
}
        
// --------------------------------------------------------------------
//  Returns the getLastError value for this system 
// --------------------------------------------------------------------
size_t System::getLastError()
{
#ifdef _WIN32
    return size_t(GetLastError());
#else
#   pragma message "WARNING getLastError not implemented for this system"
#endif

    return 0;
}
        
// --------------------------------------------------------------------
//  Returns the number of processors on the current system
// --------------------------------------------------------------------
size_t System::getNumProcessors()
{
#ifdef _WIN32
    return size_t(GetActiveProcessorCount(ALL_PROCESSOR_GROUPS));
#elif defined __linux__
    return size_t(sysconf(_SC_NPROCESSORS_CONF));
#else
#   pragma message "WARNING getNumProcessors not implemented for this system"
#endif

    return 0;
}

// --------------------------------------------------------------------
//  Returns the current process ID
// --------------------------------------------------------------------
size_t System::getCurrentPid()
{
#ifdef _WIN32
    return size_t(GetCurrentProcessId());
#elif defined __linux__
    return size_t(getpid());
#else
#   pragma message "WARNING getCurrentPid not implemented for this system"
#endif

    return 0;
}
        
// --------------------------------------------------------------------
// Returns the current thread ID
// --------------------------------------------------------------------
size_t System::getCurrentTid()
{
#ifdef _WIN32
    return size_t(GetCurrentThreadId());
#elif defined __linux__
    return size_t(gettid());
#else
#   pragma message "WARNING getCurrentTid not implemented for this system"
#endif

    return 0;
}

} // namespace vox