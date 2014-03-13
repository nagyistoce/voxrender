/* ===========================================================================
                                                                           
   Project: Volume Transform Library                                       
                                                                           
   Description: Performs volume transform operations                       
                                                                           
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
#include "Filter.h"

// Include Dependencies
#include "VoxLib/Core/Common.h"

namespace vox {
namespace volt {

class FilterManager::Impl
{
public:
    std::map<String,std::shared_ptr<Filter>> m_filters;
    FilterCallback m_callback;
    boost::mutex m_mutex;
};

// ----------------------------------------------------------------------------
//  Instantiates the pImpl member structure for the plugin manager
// ----------------------------------------------------------------------------
FilterManager & FilterManager::instance()
{ 
    static FilterManager fmanager;
    return fmanager;
}

// ----------------------------------------------------------------------------
//  Constructor
// ----------------------------------------------------------------------------
FilterManager::FilterManager() : m_pImpl(new Impl) { }

// ----------------------------------------------------------------------------
//  Registers a filter change event callback
// ----------------------------------------------------------------------------
void FilterManager::registerCallback(FilterCallback callback)
{
    m_pImpl->m_callback = callback;
}

// ----------------------------------------------------------------------------
//  Removes a filter from the filter manager
// ----------------------------------------------------------------------------
void FilterManager::remove(std::shared_ptr<Filter> filter)
{
    boost::mutex::scoped_lock lock(m_pImpl->m_mutex);
    
    auto iter = m_pImpl->m_filters.find(filter->name());
    if (iter != m_pImpl->m_filters.end() && iter->second == filter)
    {
        m_pImpl->m_filters.erase(iter);
        if (m_pImpl->m_callback) m_pImpl->m_callback(filter, false);
    }
}

// ----------------------------------------------------------------------------
//  Adds a new filter to the filter manager
// ----------------------------------------------------------------------------
void FilterManager::add(std::shared_ptr<Filter> filter)
{
    boost::mutex::scoped_lock lock(m_pImpl->m_mutex);

    if (!filter) throw Error(__FILE__, __LINE__, VOLT_LOG_CAT, 
        "Attempted to add invalid (null) filter", Error_MissingData);

    m_pImpl->m_filters.insert(std::make_pair(filter->name(), filter));

    if (m_pImpl->m_callback) m_pImpl->m_callback(filter, true);
}

// ----------------------------------------------------------------------------
//  Returns parameter information on a filter
// ----------------------------------------------------------------------------
std::shared_ptr<Filter> FilterManager::find(String const& name)
{
    boost::mutex::scoped_lock lock(m_pImpl->m_mutex);
    
    auto iter = m_pImpl->m_filters.find(name);
    if (iter != m_pImpl->m_filters.end())
    {
        return iter->second;   
    }

    throw Error(__FILE__, __LINE__, VOLT_LOG_CAT, "Filter not found", Error_BadToken);
}

// ----------------------------------------------------------------------------
//  Returns a list of the registered filtering options
// ----------------------------------------------------------------------------
void FilterManager::getFilters(std::list<std::shared_ptr<Filter>> & filters)
{
    boost::mutex::scoped_lock lock(m_pImpl->m_mutex);

    BOOST_FOREACH (auto & filter, m_pImpl->m_filters)
    {
        filters.push_back(filter.second);
    }
}

} // namespace volt
} // namespace vox