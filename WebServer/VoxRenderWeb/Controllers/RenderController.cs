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
    public class RenderController : ApiController
    {
        [HttpGet]
        public Server Open()
        {
            return new Server
            {
                host = Request.RequestUri.Host.ToString() + ":8000",
                key = "0"
            };
        }
    }
}
