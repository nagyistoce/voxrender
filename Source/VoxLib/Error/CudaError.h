/* ===========================================================================

	Project: VoxRender - CudaError

	Defines an exception class which is thrown to indicate that an internal
    call to the Cuda runtime library has failed. The associated CudaError and
    message are provided.

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
#ifndef VOX_CUDA_ERROR
#define VOX_CUDA_ERROR

// Include Dependencies
#include "VoxLib/Core/CudaCommon.h"
#include "VoxLib/Core/Types.h"
#include "VoxLib/Error/Error.h"
#include "VoxLib/Error/ErrorCodes.h"

// API namespace
namespace vox
{

    /**
     * VoxRender's Cuda Runtime Error Exception Class
     *
     * This exception is thrown by the library when an error occurs 
     * while making a CUDA library call.
     */ 
    class VOX_EXPORT CudaError : public Error
    {
    public:
        /** 
         * Constructor - Initializes the exception content
         *
         * @param _file     The file from which the error was thrown (use the __FILE__ macro)
         * @param _line     The line from which the error was thrown (use the __LINE__ macro)
         * @param _category The system associated with the exception (internally Vox)
         * @param _msg      The error message to be returned when what() is called.
         * @param _code     The CudaError value associated with the Error
         * @param _device   The ID of the device associated with the Error (negative if none)
         */
        CudaError(const char* _file, int _line, const char* _category, 
                  String const& _msg, cudaError _code, int _device = -1) : 
          Error(_file, _line, _category, _msg, Error_System), 
          device(_device), cudaCode(_code)
        {  
            cudaMessage = cudaGetErrorString( cudaCode );
            message += ": ";
            message += cudaMessage;
        }

        /** 
         * Verifies the specified cuda error code implies success 
         * 
         * This function check is the error code passed in is success.
         * If not, a CudaError object is constructed on thrown based
         * on the input parameters
         *
         * @param file     The file from which the check is issued (use the __FILE__ macro)
         * @param line     The line from which the check is issued (use the __LINE__ macro)
         * @param category The system associated with the exception 
         * @param code     The CudaError value to check for an error
         * @param device   The ID of the device associated with the error
         */
        static void check(const char* file, int line, const char* category,
                          char const* method, cudaError_t const& errorId,
                          int device = -1)
        {
		    if (errorId != cudaSuccess) throw CudaError(file, line, category, method, errorId, device);
        }

        cudaError   cudaCode;    ///< The CUDA error code 
        const char* cudaMessage; ///< System error message
        int         device;      ///< The CUDA device ID (provided by VoxRender)
    };

}

// CUDA Return Value Error to Exception Macro
#define VOX_CUDA_CHECK(X) { CudaError::check(__FILE__, __LINE__, "VOX", #X, X); }

// End definition
#endif // VOX_CUDA_ERROR