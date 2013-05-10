/* ===========================================================================

	Project: VoxLib - Bitmap

	Description: Defines a bitmap class for image import/export operations

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
#ifndef VOX_BITMAP
#define VOX_BITMAP

// Include Dependencies
#include "VoxLib/Core/CudaCommon.h"
#include "VoxLib/IO/OptionSet.h"
#include "VoxLib/IO/Resource.h"

// API namespace
namespace vox
{

class VOX_EXPORT Bitmap;

typedef std::function<Bitmap(ResourceIStream & data, OptionSet const& options)> BitmapImporter;

typedef std::function<void(ResourceOStream & data, OptionSet const& options, Bitmap const& bitmap)> BitmapExporter;


/** Generic bitmap class */
class VOX_EXPORT Bitmap
{
public:
    /** Bitmap data format flags */
    enum Type
    {
        Type_Begin,                  ///< Begin iterator for Type enumeration
        Type_A8R8G8B8 = Type_Begin,  ///< 8 bit per channel ARGB
        Type_R8G8B8,                 ///< 8 bit per channel RGB
        Type_Gray8,                  ///< 8 bit grayscale
        Type_Gray12,                 ///< 12 bit grayscale
        Type_Gray16,                 ///< 16 bit grayscale
        Type_End                     ///< End iterator for Type enumeration
    };

    /** Printable strings for Type enum */
    static String typeStr[Type_End];

public:
	inline static Bitmap imprt(ResourceId const& identifier, OptionSet const& options = OptionSet())
    {
        return imprt(ResourceIStream(identifier), options);
    }

	static Bitmap imprt(ResourceIStream & data, 
                        OptionSet const&  options   = OptionSet(),
                        String const&     extension = String())
    {
        return Bitmap();
    }

	inline void exprt(ResourceId const& identifier, OptionSet const& options = OptionSet()) const
    {
        return exprt(ResourceOStream(identifier), options);
    }

	void exprt(ResourceOStream & data, 
                OptionSet const&  options   = OptionSet(), 
                String const&     extension = String()) const
    {
    }

    //static void registerImportModule(String const& extension, BitmapImporter importer) { }

    //static void registerExportModule(String const& extension, BitmapExporter exporter) { }
};

} // namespace vox

// End definition
#endif // VOX_BITMAP