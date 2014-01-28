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
#ifndef VOX_VOLT_VERSION_H
#define VOX_VOLT_VERSION_H

// Stringify macro
#define VOX_VOLT_XSTR(v) #v
#define VOX_VOLT_STR(v) VOX_VOLT_XSTR(v)

// Plugin version info
#define VOX_VOLT_VERSION_MAJOR 1
#define VOX_VOLT_VERSION_MINOR 0
#define VOX_VOLT_VERSION_PATCH 0

// API support version info
#define VOX_VOLT_API_VERSION_MIN_STR "0.0.0"
#define VOX_VOLT_API_VERSION_MAX_STR "999.999.999"

// Plugin version string
#define VOX_VOLT_VERSION_STRING VOX_VOLT_STR(VOX_VOLT_VERSION_MAJOR) \
	"." VOX_VOLT_STR(VOX_VOLT_VERSION_MINOR) "." VOX_VOLT_STR(VOX_VOLT_VERSION_PATCH)

// End definition
#endif // VOX_VOLT_VERSION_H