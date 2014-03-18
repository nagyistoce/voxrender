/* ===========================================================================

	Project: Standard Volume Filters
    
	Description: Exposes some standard filters provided by the volt library

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
#include "Filters.h"

// API dependencies
#include "VoxLib/Core/Common.h"
#include "VoxLib/Core/Format.h"

namespace vox {

using namespace volt;

namespace {
namespace filescope {

    // Parameter generation for non-floating point types
    template<typename T>
    void generateLinearParams(Scene const& scene, std::list<FilterParam> & params)
    {
        Vector<double,2> range(std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

        // Shift/Scale parameter ranges
        params.push_back(FilterParam("Shift", FilterParam::Type_Float, "0", 
            boost::lexical_cast<String>(range)));
        params.push_back(FilterParam("Scale", FilterParam::Type_Float, "1", 
            boost::lexical_cast<String>(range)));

        // Scale inversion toggle
        params.push_back(FilterParam("Invert", FilterParam::Type_Check, "Invert the scale factor?", ""));

        // Output volume format parameter
        auto defaultType = Volume::typeToString(scene.volume->type());
        String types;
        for (int i = Volume::Type_Begin; i < Volume::Type_End; ++i)
        {
            types += Volume::typeToString((Volume::Type)i);
            if (i != Volume::Type_End - 1) types += ",";
        }

        params.push_back(volt::FilterParam("Type", volt::FilterParam::Type_Radio, defaultType, types));

        // Current value range of volume data
        auto unormRange = scene.volume->valueRange();
        unormRange[0] = (unormRange[0] < 0) ? abs(unormRange[0]) * range[0] : unormRange[0] * range[1];
        unormRange[1] = (unormRange[1] < 0) ? abs(unormRange[1]) * range[0] : unormRange[1] * range[1];
        String rangeStr = boost::lexical_cast<String>(floor(unormRange[0])) + " to " +
                          boost::lexical_cast<String>(floor(unormRange[1]));
        params.push_back(volt::FilterParam("Range", volt::FilterParam::Type_Label, rangeStr, "")); 
    }

    // Parameter generation for floating point types
    // :TODO:

} // namespace filescope
} // namespace anonymous

// ----------------------------------------------------------------------------
//  Generates the filter parameters for a scale/shift operation
// ----------------------------------------------------------------------------
void LinearFilter::getParams(Scene const& scene, std::list<volt::FilterParam> & params)
{
    // Shift/Scale/Current Range value parameters
    switch (scene.volume->type())
    { 
    case Volume::Type_UInt8:   filescope::generateLinearParams<UInt8>  (scene, params); break;
    case Volume::Type_UInt16:  filescope::generateLinearParams<UInt16> (scene, params); break;
    case Volume::Type_UInt32:  filescope::generateLinearParams<UInt32> (scene, params); break;
    case Volume::Type_Int8:    filescope::generateLinearParams<Int8>   (scene, params); break;
    case Volume::Type_Int16:   filescope::generateLinearParams<Int16>  (scene, params); break;
    case Volume::Type_Int32:   filescope::generateLinearParams<Int32>  (scene, params); break;
    default: break;
    }
}

} // namespace vox