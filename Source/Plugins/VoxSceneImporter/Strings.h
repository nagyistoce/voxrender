/* ===========================================================================

    Project: Vox Scene Importer - Module definition for scene importer

    Description: Defines the string constants for the XML format scene files

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

// Version

auto V_MAJOR = "Major"; ///< Major version number
auto V_MINOR = "Minor"; ///< Minor version number

// Volume

auto V_SPACING = "Spacing";
auto V_OFFSET  = "Offset";

// Camera

auto C_APERTURE   = "ApertureSize";     ///< Aperture size (mm)
auto C_FOV        = "FieldOfView";      ///< Field of view (degrees)
auto C_FOCAL_DIST = "FocalDistance";    ///< Focal distance (mm)
auto C_POSITION   = "Position";         ///< Camera position (mm X mm X mm)
auto C_TARGET     = "Target";           ///< Camera target view position (mm X mm X mm)
auto C_FWIDTH     = "FilmWidth";        ///< Film width (pixels)
auto C_FHEIGHT    = "FilmHeight";       ///< Film height (pixels)
auto C_UP         = "Up";               ///< Camera up vector (mm X mm X mm)
auto C_RIGHT      = "Right";            ///< Camera right vector (mm X mm X mm)
auto C_EYE        = "Eye";              ///< Camera eye vector (mm X mm X mm)

// Transfer

auto T_RESOLUTION = "Resolution";   ///< Transfer map resolution (pixels X pixels X pixels)

// |-> Materials

auto M_GLOSSINESS = "Glossiness";       ///< Material specularity
auto M_THICKNESS  = "Thickness";        ///< Optical thickness (/mm)

// |-> Node

// Settings

auto P_STEP_PRIMARY    = "PrimaryStepSize"; ///< Primary sample step size (mm)
auto P_STEP_SHADOW     = "ShadowStepSize";  ///< Shadow sample step size (mm)
auto P_STEP_OCCLUDE    = "OccludeStepSize"; ///< Occlude sample step size (mm)
auto P_SAMPLES_OCCLUDE = "OccludeSamples";  ///< Occlude Samples (count)