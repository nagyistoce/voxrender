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

// Begin definition
#ifndef VOX_VOLT_FILTER_H
#define VOX_VOLT_FILTER_H

// Include Dependencies
#include "VoxVolt/Common.h"
#include "VoxLib/Core/CudaCommon.h"
#include "VoxScene/Scene.h"

// API namespace
namespace vox {
namespace volt {

/** Filter parameter */
struct VOX_VOLT_EXPORT FilterParam
{
    enum Type
    {
        Type_Begin,
        Type_Float = Type_Begin,
        Type_Int,
        Type_Radio,
        Type_End
    };

    FilterParam(String const& p_name, Type p_type, String const& p_value, String const& p_range) :
        name(p_name), type(p_type), value(p_value), range(p_range)
    {
    }

    String name;  ///< Display Name
    Type   type;  ///< Data Type
    String value; ///< Default Value
    String range; ///< Value range (radio = [Opt1 Opt2 Opt3 ... ])
};

/** Filtering class */
class VOX_VOLT_EXPORT Filter
{
public:
    /** Returns the name of this filter, ie "Type.Sub_Type.Name" */
    virtual String name() = 0; 

    /** Generates a list of the parameters for this filter */
    virtual void getParams(std::list<FilterParam> & params) = 0;

    /** Performs the function of this filter on the scene */
    virtual void execute(Scene & scene, OptionSet const& params) = 0;
};

/** Implements transforms for convolution operations */
class VOX_VOLT_EXPORT FilterManager
{
public:
    /** Returns the global FilterManager */
    static FilterManager & instance();

    /** Registers a callback for filter change events */
    void registerCallback(std::function<void()> callback);

    /** Returns parameter information on a filter */
    std::shared_ptr<Filter> find(String const& name);
    
    /** Adds a new filter to the manager */
    void add(std::shared_ptr<Filter> filter);

    /** Removes a filter from the manager */
    void remove(std::shared_ptr<Filter> filter);

    /** Generates a list of the available filters in the manager */
    void getFilters(std::list<std::shared_ptr<Filter>> & filters);

private:
    FilterManager();

    class Impl;
    Impl * m_pImpl;
};

} // namespace volt
} // namespace vox

// End definition
#endif // VOX_VOLT_FILTER_H