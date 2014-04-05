using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Runtime.InteropServices;

namespace FundusWeb.Models
{
    // Image Rendering Service
    public class VoxServer
    {
        [DllImport("VoxServer.dll", EntryPoint = "startup", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        public static extern int startup(String directory);

        [DllImport("VoxServer.dll", EntryPoint = "render", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        public static extern int render();

        [DllImport("VoxServer.dll", EntryPoint = "version", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        public static extern String version();

        [DllImport("VoxServer.dll", EntryPoint = "shutdown", CallingConvention = CallingConvention.Cdecl)]
        public static extern void shutdown();
    }
}