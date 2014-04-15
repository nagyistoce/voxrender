using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Runtime.InteropServices;

namespace FundusWeb.Models
{
    // Image Rendering Service
    public class ServerLib
    {
        [DllImport("VoxServer.dll", EntryPoint = "voxServerStart", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        public static extern int voxServerStart(String directory);

        [DllImport("VoxServer.dll", EntryPoint = "render", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        public static extern int render();

        [DllImport("VoxServer.dll", EntryPoint = "voxServerVersion", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        public static extern String voxServerVersion();

        [DllImport("VoxServer.dll", EntryPoint = "voxServerEnd", CallingConvention = CallingConvention.Cdecl)]
        public static extern void voxServerEnd();
    }
}