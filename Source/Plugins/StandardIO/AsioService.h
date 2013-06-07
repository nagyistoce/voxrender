/* ===========================================================================

    Project: AsioService

    Description: A foundation Async IO service for developing WebClient APIs

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
#ifndef ASIO_ASIO_SERVICE_H
#define ASIO_ASIO_SERVICE_H

// Include Dependencies
#include "StandardIO/Common.h"
#include "VoxLib/Core/Functors.h"
#include "VoxLib/IO/Resource.h"
#include "VoxLib/IO/ResourceModule.h"

// Boost Library Dependencies
#include <boost/enable_shared_from_this.hpp>
#include <boost/thread.hpp>

namespace vox {

class AsioService;

/**
 * Request interface used by the AsioService 
 *
 * This interface provides a low level method of sending requests to the AsioService.
 * Libraries that provide more specific libraries on top of this (HttpClient, FtpClient, StandardIO module)
 * will implement classes which derive from or utilize the AsioRequest to complete their requests.
 */
class AsioRequest : public std::enable_shared_from_this<AsioRequest>
{
public:
    /** Initializes a new async request */
    AsioRequest(
        ResourceId &     identifier, ///< The resource identifier
        OptionSet const& options     ///< The advanced access options
        );

    /** Ensures request cancellation before shutdown */
    ~AsioRequest();

    /** Detaches the request for asynchronous cancellation */
    void detach();

    /** Returns the curl handle associated with the request */
    void * handle() { return m_handle; }

    /** Called upon completion of a request */
    virtual void complete(std::exception_ptr ex) = 0;

    /** Calls completion and then performs cleanup */
    void onComplete(std::exception_ptr ex);

    /** Issues the request to the asio service */
    void issueRequest();

protected:
    friend AsioService;    

    std::shared_ptr<AsioRequest> m_self;

    boost::mutex m_mutex;

    void * m_handle; ///< Internal request handle
};

} // namespace vox

// End definition
#endif // ASIO_ASIO_SERVICE_H
