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

// Begin definition
#ifndef VOX_SERVER_OP_CODES_H
#define VOX_SERVER_OP_CODES_H

namespace vox {

enum OpCode 
{
    OpCode_Error     = 0x00,    ///< Error message [S->C]
    OpCode_BegStream = 0x01,    ///< Begin render stream [C->S]
    OpCode_EndStream = 0x02,    ///< Terminate render stream [C->S]
    OpCode_Status    = 0x03,    ///< Render status packet [S->C]
    OpCode_StatusReq = 0x04,    ///< Request for render status packet to be sent [C->S]
    OpCode_DirList   = 0x05,    ///< Contains a listing of scene file names from the rootDir [S->C]
    OpCode_Update    = 0x06,    ///< Make change to current render scene [C->S]
    OpCode_SceneReq  = 0x07,    ///< Send XML scene file back to client [S->C]
    OpCode_Scene     = 0x08,    ///< Scene data file [S<->C]
    OpCode_Frame     = 0x09     ///< Render frame [S->C]
};

} // namespace vox

// End definition
#endif // VOX_SERVER_OP_CODES_H