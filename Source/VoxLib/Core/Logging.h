/* ===========================================================================

	Project: VoxRender - Logging interface

	Description: Implements a front-end for run-time logging

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

=========================================================================== */

// Begin definition
#ifndef VOX_LOGGING_H
#define VOX_LOGGING_H

// Include Dependencies
#include "VoxLib/Core/CudaCommon.h"
#include "VoxLib/Error/Error.h"

// Boost Dependencies
#include <boost/thread.hpp>

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
        Severity_Trace      = 100, ///< Program trace points
        Severity_Trace1     =  99, ///< Program trace points (2nd level)
        Severity_Trace2     =  98, ///< Program trace points (3rd level)
		Severity_Debug      = 200, ///< Developer information 
		Severity_Debug2     = 199, ///< Developer information (2nd level)
		Severity_Debug3     = 198, ///< Developer information (3rd level)
		Severity_Info       = 300, ///< Status information
		Severity_Warning    = 400, ///< Something that may be an issue
		Severity_Error      = 500, ///< Unable to complete the operation
		Severity_Fatal      = 600  ///< The error is likely fatal
	};

	/** 
	 * VoxRender logging interface
     * 
     * The logger class interface allows user access to the VoxLib logging system. The base class 
     * associated with logging is the Logger class. In addition to containing the members which modify 
     * the log filtering level, the logger backend can be replaced with a user defined handler function
     * through the interface provided here. 
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
     * \section Filtering
     *
     * The logging system provides functionality for controlling the log entries
     * which are filtered out during runtime.
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
         * @param file     The file from which the log entry was made (use the __FILE__ macro)
         * @param line     The line from which the error was made (use the __LINE__ macro)
         * @param category The system associated with the exception (internally Vox)
         * @param code     The error code associated with the Error
         * @param msg      The message to be entered in the log stream
		 */
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
		inline static void setHandler(vox::ErrorHandler errorHandler) { m_errorHandler = errorHandler; }

		/** 
		 * Sets the error handler used by the logging system.
		 *
		 * @return integer code corresponding to the last error logged
		 */
		inline static int getLastError() { return m_lastError; }

		/** 
		 * Sets the filtering level of the logging system. If you wish to 
         * filter user code messages as well, you must use the VOX_LOG_... format
         * macros which operate filtering efficiently.
		 *
		 * @param filter The minumum severity level to log
		 */
		inline static void setFilteringLevel(int filter) { m_filter = filter; }

		/** 
		 * Adds an additional category to the list of filtered log categories.
         *
         * Filtered log categories are category strings which will be ignored
         * when determining whether to dispatch a log entry to the backend.
		 *
		 * @param category The name of the category
		 */
        static void addCategoryFilter(String const& category)
        {
            boost::mutex::scoped_lock lock(m_catMutex);

            m_catFilters.push_back(category);
        }

		/** 
		 * Removes a category from the list of filtered log categories
         *
         * Filtered log categories are category strings which will be ignored
         * when determining whether to dispatch a log entry to the backend.
		 *
		 * @param category The name of the category
		 */
        static void removeCategoryFilter(String const& category)
        {
            boost::mutex::scoped_lock lock(m_catMutex);

            m_catFilters.remove(category);
        }

		/** 
		 * Returns true if a category is in the list of filtered categories
         *
         * Filtered log categories are category strings which will be ignored
         * when determining whether to dispatch a log entry to the backend.
		 *
		 * @param category The name of the category
		 */
        static bool isCategoryFiltered(String const& category)
        {
            return std::find(m_catFilters.begin(), m_catFilters.end(), category) != m_catFilters.end();
        }

		/** 
		 * Gets the filtering level of the logging system.
		 *
		 * @return The minumum severity level being logged
		 */
		inline static int getFilteringLevel() { return m_filter; }

	private:
        static std::list<String> m_catFilters; ///< Filtered log categories
        static boost::mutex      m_catMutex;   ///< Cat filters mutex 

		static ErrorHandler m_errorHandler;	///< Error handler for logging
		static int          m_filter;		///< Filter level for log entries
		static int          m_lastError;	///< Code of last error logged
	};

    // Logging macro incorperating user defined severity levels
    /// :TODO: Check category map
#define VOX_LOG(SEV, CODE, CAT, MSG)                                    \
    if (vox::Logger::getFilteringLevel() <= SEV)                        \
    {                                                                   \
        vox::Logger::addEntry(SEV, CODE, CAT, MSG, __FILE__, __LINE__); \
    }

    // Logging macros for api specific filter levels
#define VOX_LOG_TRACE(CAT, MSG)                                                                    \
    if (vox::Logger::getFilteringLevel() <= vox::Severity_Trace)                                   \
    {                                                                                              \
        vox::Logger::addEntry(vox::Severity_Trace, vox::Error_None, CAT, MSG, __FILE__, __LINE__); \
    }
#define VOX_LOG_INFO(CAT, MSG)                                                                    \
    if (vox::Logger::getFilteringLevel() <= vox::Severity_Info)                                   \
    {                                                                                             \
        vox::Logger::addEntry(vox::Severity_Info, vox::Error_None, CAT, MSG, __FILE__, __LINE__); \
    }
#define VOX_LOG_DEBUG(CAT, MSG)                                                                    \
    if (vox::Logger::getFilteringLevel() <= vox::Severity_Debug)                                   \
    {                                                                                              \
        vox::Logger::addEntry(vox::Severity_Debug, vox::Error_None, CAT, MSG, __FILE__, __LINE__); \
    }
#define VOX_LOG_WARNING(CODE, CAT, MSG)                                                              \
    if (vox::Logger::getFilteringLevel() <= vox::Severity_Warning)                                   \
    {                                                                                                \
        vox::Logger::addEntry(vox::Severity_Warning, CODE, CAT, MSG, __FILE__, __LINE__);            \
    } 
#define VOX_LOG_ERROR(CODE, CAT, MSG)                                                   \
    if (vox::Logger::getFilteringLevel() <= vox::Severity_Error)                        \
    {                                                                                   \
        vox::Logger::addEntry(vox::Severity_Error, CODE, CAT, MSG, __FILE__, __LINE__); \
    } 
#define VOX_LOG_EXCEPTION(SEV, EXC)                                                                  \
    if (vox::Logger::getFilteringLevel() <= SEV)                                                     \
    {                                                                                                \
        vox::Logger::addEntry(EXC, SEV);                                                             \
    } 

} // namespace vox

#endif // VOX_LOGGING_H