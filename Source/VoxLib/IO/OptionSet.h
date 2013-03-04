/* ===========================================================================

	Project: Uniform Resource IO 
    
	Description: Defines an option set class for IO option specifications

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
#ifndef VOX_OPTION_SET_H
#define VOX_OPTION_SET_H

// Include Dependencies
#include "VoxLib/Core/CudaCommon.h"
#include "VoxLib/Core/Format.h"
#include "VoxLib/Error/Error.h"
#include "VoxLib/Error/ErrorCodes.h"

// API Namespace
namespace vox
{

/** Extended type-casting class inheriting from multi-map */
// Developer note: DO NOT add member variables here, it inherits STL
class OptionSet : public std::multimap<std::string, std::string>
{
public:
    /** Map style access operator - returns value of one matched key */
    inline std::string const& operator[] (std::string const& key) const
    {
        return lookup(key);
    }

    /** Type-cast assisted insertion functionality */
    template<typename T>
    void addOption(std::string const& key, T const& val)
    {
        try
        {
            insert(
                std::make_pair(key, boost::lexical_cast<std::string>(val))
                );
        }
        catch (boost::bad_lexical_cast &)
        {
            std::string const err = "";
            throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY, err, Error_BadToken);
        }
    }

    /** Wrapper for clarity in option set add operations */
    void addOption(std::string const& key, std::string const& val)
    {
        insert(std::make_pair(key, val));
    }

    /** Type-cast assisted find functionality */
    template<typename T>
    T lookup(std::string const& key, T const& def) const
    {
        auto const& iter = find(key);
        if (iter != end())
        {
            try { return boost::lexical_cast<T>(iter->second); }
            catch (boost::bad_lexical_cast &)
            {
                std::string const err = "";
                throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY, err, Error_BadToken);
            }
        }
        else
        {
            return def;
        }
    }

    /** Type-cast assisted find functionality */
    template<typename T>
    T lookup(std::string const& key) const
    {
        auto const& iter = find(key);
        if (iter != end())
        {
            try { return boost::lexical_cast<T>(iter->second); }
            catch (boost::bad_lexical_cast &)
            {
                std::string const err = "Value could not be cast to type T:TODO:";
                throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY, err, Error_BadToken);
            }
        }
        else
        {
            throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY,
                       "Missing Key", Error_NotFound);
        }
    }

    /** Overload for string lookup */
    std::string const& lookup(std::string const& key, std::string const& def) const
    {
        auto const& iter = find(key);
        if (iter != end())
        {
            return iter->second;
        }
        else
        {
            return def;
        }
    }

    /** Overload for string lookup */
    std::string const& lookup(std::string const& key) const
    {
        auto const& iter = find(key);
        if (iter != end())
        {
            return iter->second;
        }
        else
        {
            throw Error(__FILE__, __LINE__, VOX_LOG_CATEGORY,
                        "Key not found", Error_NotFound);
        }
    }
};

} // namespace vox

// End definition
#endif // VOX_OPTION_SET