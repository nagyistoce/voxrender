using FundusWeb.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Web.Http;
using System.IO;
using System.Text.RegularExpressions;
using System.Diagnostics;
using System.Web;
using System.Runtime.InteropServices;

namespace FundusWeb.Controllers
{
    public class SceneController : ApiController
    {
        public Scene[] Get(string id)
        {
            // Query the users stored scene files
            var user = "sherm190";
            var filePath = HttpContext.Current.Server.MapPath("~/Users/" + user + "/");

            return new Scene[]{ new Scene() };
        }

        public void Render(string id)
        {
            // Generate the expected filepath to the resource on disk
            var filePath = HttpContext.Current.Server.MapPath("~/Users/" + id);

           // Begin rendering the scene file
        }

        public Scene Post(Scene scene)
        {
            // Ensure that the specified scene does not already exist
            var user     = "sherm190";
            var filePath = HttpContext.Current.Server.MapPath("~/Users/" + user + "/" + scene.id);

            // Parse the data URL to extract the actual binary file 
            var regex    = new Regex(@"data:(?<mime>[\w/]+);(?<encoding>\w+),(?<data>.*)", RegexOptions.Compiled);
            var match    = regex.Match(scene.data);
            var mime     = match.Groups["mime"].Value;
            var encoding = match.Groups["encoding"].Value;
            var data     = match.Groups["data"].Value;
            var ibytes   = Convert.FromBase64String(data);

            // Write The file the the users data space
            try
            {
                FileStream stream = new FileStream(filePath, FileMode.CreateNew);
                stream.Write(ibytes, 0, ibytes.Length);
                stream.Close();
            }
            catch (IOException e)
            {
                return new Scene { id = "123", data = null };
            }
            
            return new Scene { id = "789", data = null };
        }

        public bool Delete(string id)
        {
            var filePath = HttpContext.Current.Server.MapPath("~/Users/" + id);

            try
            {
                File.Delete(filePath);
            }
            catch (Exception e) { return false; }

            return true;
        }
    }
}