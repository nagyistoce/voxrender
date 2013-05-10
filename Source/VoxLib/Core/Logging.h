/* ===========================================================================

	Project: VoxRender - Logging interface

	Description: Implements a front-end for run-time logging

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

// Begin definition
#ifndef VOX_LOGGING_H
#define VOX_LOGGING_H

// Include Dependencies
#include "VoxLib/Core/CudaCommon.h"
#include "VoxLib/Error/Error.h"

// API namespace
namespace vox
{
    class VOX_EXPORT Logger; // Class prototype

	/** Vox Error Handler : Abort if the error severity exceeds Severity_Warning */
	VOX_EXPORT void ErrorAbort(char const* file, int line, int severity, int code,
        char const* category, char const* message);

	/** Vox Error Handler : Ignore any errors */
    VOX_EXPORT void ErrorIgnore(char const* file, int line, int severity, int code,
        char const* category, char const* message);

	/** 
     * Vox Error Handler : Dispatches error messages to std::clog
     *
     * The format of the output log entries are as follows:
     *  [ MM-DD-YY HH::MM::SS UTC SeverityLevel ErrorCode ] pid tid category message <file="file",line=#>\n
     */
	VOX_EXPORT void ErrorPrint(char const* file, int line, int severity, int code,
        char const* category, char const* message);

	/** Error handling callback function */
	typedef std::function<void(char const* file, int line, int severity, int code,
        char const* category, char const* message)> ErrorHandler;
 
	/** API error severity levels */
	enum Severity
	{
        Severity_Trace      =   0, ///< Program trace points
		Severity_Debug      = 100, ///< Developer information 
		Severity_Info       = 200, ///< Status information
		Severity_Warning    = 300, ///< Something that may be an issue
		Severity_Error      = 400, ///< Unable to complete the operation
		Severity_Fatal      = 500  ///< The error is likely fatal
	};

	/** 
	 * VoxRender log entry
     * 
     * The log entry class derives from std::ostringstream and allows 
     * for message content to be appended to the body of the log before
     * it is sent to the backend.
	 */
	class VOX_EXPORT LogEntry : public std::ostringstream
    {
        friend Logger;

    public:
        ~LogEntry();

        LogEntry(LogEntry&& original) :
            std::ostringstream(std::move(original))
        {
            m_severity = original.m_severity;
            m_code     = original.m_code;
            m_category = original.m_category;
            m_file     = original.m_file;
            m_line     = original.m_line;
        }
        
    private:
        LogEntry(int severity, int code, char const* category, 
                 char const* file, int line) :
          m_severity(severity), m_code(code), m_line(line), m_file(file),
          m_category(category)
        {
        }

        LogEntry(LogEntry&) { } // Privatize copy constructor

		int         m_severity; ///< Severity of associated error
        int         m_code;     ///< Code number of associated error
        char const* m_category; ///< Log entry category option
        char const* m_file;     ///< Optional filename
        int         m_line;     ///< Optional line number
    };

	/** 
	 * VoxRender logging interface
     * 
     * The logger class interface allows user access to the VoxRender logging system. The base class 
     * associated with logging is the Logger class. In addition to containing the members which modify 
     * the log filtering level, the logger backend can be replaced with a user defined handler function.
     * 
     * \section Logger Backends
     * The logger backend is a function which takes in the log entry content as parameters and performs 
     * the desired logging operation with them. Several log backends are provided internally:
     *
     *  - \p ErrorPrint
     *  - \p ErrorAbort
     *  - \p ErrorIgnore
     *
     * And the respective documentation provides descriptions of their function. By default, the ErrorPrint 
     * backend will be used. The ErrorPrint function is defined for general use, and outputs formatted 
     * error messages to the std::cerr stream. If logging to a file is desired, the user can simply redirect 
     * the stream to the desired filebuf using the std::stream::set_rdbuf and std::stream::rdbuf member 
     * functions.
     *
     * \section Example Usage
     * Log entries can make their way to the log backend in one of two ways. The first method is by using 
     * the Logger class's addEntry members. The addEntry functions, which is overloaded for use with the 
     * internal exception object Error, immediately sends the log entry to the backend for processing. The 
     * only overhead is in the filtering process, described in the next section. An example of this usage 
     * is provided below:
     *
     * \code
     *  // Logging an exception object
     *  try { execute some code...
     *  } catch( Error const& error )
     *  {
     *      vox::Logger::addEntry( error, Severity_Error );
     *  }
     *
     *  /// Logging an info string
     *  vox::Logger::addEntry( Severity_Info, Error_None, "category",
     *      "error message", __FILE__, __LINE__ );
     * \endcode
     *
     * The other logging method allows the message portion of the log entry to be provided to a given stream 
     * object following the initial addEntry call. The actual request will be logged as soon as the Log object 
     * goes out of scope. In the case of cascading log entries in a single functional unit of code, it may be 
     * benificial to wrap the log calls in a seperate { } block to ensure the entries are dispatched in the 
     * order they are constructed. The same overloaded member function (addEntry) is used as before, but now 
     * the  message parameter is left out and provided to the stream. An example usage follows:
     *
     * \code
     *
     *  /// Logging an info string
     *  char const* myCategory = "category";
     * 
     *  auto logEntry = vox::Logger::addEntry( 
     *      Severity_Info, Error_None, 
     *      category, __FILE__, __LINE__); 
     *
     *  foreach(auto x, list) entry << x;
     * 
     * \endcode
     *
     * \section Filtering
     *
     * The logging system provides functionality for controlling the log entries
     * which are filtered out from processing both during runtime and during 
     * compilation. 
	 */
	class VOX_EXPORT Logger
	{
	public:
        /**
         * Sends the input log entry information to the backend
         *
         * This function bypasses the message composition stage of the
         * log entry process and sends the entry directly to the backend.
         *
         * @param file     The file from which the log entry was made
         * @param line     The line from which the error was made 
         * @param category The system associated with the exception 
         * @param message  The error message associated with the log entry
         * @param code     The error code associated with the entry
         * @param severity The severity code for the log entry
         */
        static inline void addEntry(int severity, int code, char const* category, 
            char const* message, char const* file = "", int line = 0)
        {
            m_errorHandler(file, line, severity, code, category, message);

			if (severity > Severity_Warning) m_lastError = code; 
        }

        /**
         * Adds a new log entry for a standard Error object
         *
         * @param error    The exception containing the log entry information
         * @param severity The severity code for the log entry
         */
        static inline void addEntry(Error const& error, int severity = Severity_Error)
        {
            addEntry(severity, error.code, error.category, 
                     error.message.c_str(), error.file, error.line);
        }

		/** 
		 * Adds a new log entry 
         *
         * The new log entry will be made when the LogEntry instance goes out of scope.
         *
         * @param file     The file from which the log entry was made (use the __FILE__ macro)
         * @param line     The line from which the error was made (use the __LINE__ macro)
         * @param category The system associated with the exception (internally Vox)
         * @param code     The error code associated with the Error
		 */
		static inline LogEntry addEntry(int severity, int code, 
            char const* category, char const* file = "", int line = 0)
        {
            return LogEntry(severity, code, category, file, line);
        }

        /** String based overload for addEntry */
        static inline void addEntry(int severity, int code, char const* category, 
            std::string const& message, char const* file = "", int line = 0)
        {
            m_errorHandler(file, line, severity, code, category, message.c_str());

			if (severity > Severity_Warning) m_lastError = code; 
        }

		/** 
		 * Sets the error handler used by the logging system.
		 *
		 * @param errorHandler The backend function to be called for logging
		 */
		inline static void setHandler(vox::ErrorHandler errorHandler) 
			{ m_errorHandler = errorHandler; }

		/** 
		 * Sets the error handler used by the logging system.
		 *
		 * @return integer code corresponding to the last error logged
		 */
		inline static int getLastError() { return m_lastError; }

		/** 
		 * Sets the filtering level of the logging system.
		 *
		 * @param filter The minumum severity level to log
		 */
		inline static void setFilteringLevel(int filter) { m_filter = filter; }

		/** 
		 * Gets the filtering level of the logging system.
		 *
		 * @return The minumum severity level being logged
		 */
		inline static int getFilteringLevel() { return m_filter; }

	private:
		static ErrorHandler m_errorHandler;	///< Error handler for logging
		static int          m_filter;		///< Filter level for log entries
		static int          m_lastError;	///< Code of last error logged
	};

    // Logging macro for performing filtered logging
#define VOX_LOGF(SEV, CODE, CAT, MSG)                                   \
    if (vox::Logger::getFilteringLevel() <= SEV)                        \
    {                                                                   \
        vox::Logger::addEntry(SEV, CODE, CAT, MSG, __FILE__, __LINE__); \
    }

} // namespace vox

#endif // VOX_LOGGING_H