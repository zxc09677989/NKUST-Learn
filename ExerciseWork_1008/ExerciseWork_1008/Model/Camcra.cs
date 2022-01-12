using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

using System.Threading.Tasks;

namespace ExerciseWork_1008.Model
{
    public class Camcra
    {
        //分局,派出所,警編號,位置
        public string 分局 { get; set; }
        public string 派出所 { get; set; }
        public string 警編號 { get; set; }
        public string 位置 { get; set; }
        public Dictionary<string, string> Datas { get; set; }
        public HashSet<string> Keys { get; set; }
        public Camcra()
        {
        this.Datas = new Dictionary<string, string>();
        }
    }
}