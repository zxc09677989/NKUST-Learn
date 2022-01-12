using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

using System.Threading.Tasks;

namespace ExerciseWork_1008.Unit
{
    public static class Filepath 
    {
        public static string GetFullepath(string filename)
        {
            return System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "App_Data", filename);
        }
    }
}