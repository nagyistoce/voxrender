/* ===========================================================================

	Project: VoxServer

	Description: Implements a WebSocket based server for interactive rendering

    Copyright (C) 2014 Lucas Sherman

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
#include "Base64.h"

// Include Dependencies
#include <boost/archive/iterators/base64_from_binary.hpp>
#include <boost/archive/iterators/binary_from_base64.hpp>
#include <boost/archive/iterators/insert_linebreaks.hpp>
#include <boost/archive/iterators/transform_width.hpp>
#include <boost/archive/iterators/ostream_iterator.hpp>

namespace vox {

namespace bai = boost::archive::iterators;

namespace {
namespace filescope {

    // Padding characters for Base64 encoding
    const std::string base64Padding[] = {"", "==","="};

} // namespace filescope
} // namespace anonymous

// ----------------------------------------------------------------------------
//  Performs Base64 encoding
// ----------------------------------------------------------------------------
String Base64::encode(String const& data) 
{
    StringStream os;

    typedef bai::base64_from_binary<bai::transform_width<const char *, 6, 8> > base64_enc;

    std::copy(base64_enc(data.c_str()), base64_enc(data.c_str() + data.size()), bai::ostream_iterator<char>(os));

    os << filescope::base64Padding[data.size() % 3];

    return os.str();
}

// ----------------------------------------------------------------------------
//  Performs Base64 decoding
// ----------------------------------------------------------------------------
String Base64::decode(String const& data) 
{
    std::stringstream os;

    typedef bai::transform_width<bai::binary_from_base64<const char *>, 8, 6> base64_dec;

    auto size = data.size();
    if (size == 0) return "";

    for (int i = 1; i <= 2; i++) 
    {
        if (data.c_str()[size - 1] == '=') size--;
    }

    std::copy(base64_dec(data.c_str()), base64_dec(data.c_str() + size), bai::ostream_iterator<char>(os));

    return os.str();
}

} // namespace vox