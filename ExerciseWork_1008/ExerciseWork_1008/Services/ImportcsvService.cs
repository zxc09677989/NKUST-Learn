using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using ExerciseWork_1008.Model;
using System.Threading.Tasks;

namespace ExerciseWork_1008.sencices
{
    public class ImportcsvService 
    {
        public List<Camcra> LoadFormfile(string FilePath)
        {
            //   Func<int, string> f = (x,y) =>
            //       {
            //        return (x+y).Tostring();
            //    };
            List<Camcra> result = new List<Camcra>();

            //System.IO.FileInfo file = new System.IO.FileInfo(FilePath);

            string []lines = System.IO.File.ReadAllLines(FilePath);
            //第1種
            //for (var i = 0; i < lines.Length; i++)
            // {
            // }
            //第2種
            // foreach(var line in lines)
            //{
            //}
            //第3種
            var Camcra = lines
                .ToList()
                .Skip(1) //跳過第一行的分類標記
                .Select(x =>
                {
                    var colums = x.Split(",");
                    var item = new Camcra()
                    {
                        分局 = colums[0],
                        派出所 = colums[1],
                        警編號 = colums[2],
                        //位置 = colums[3],
                    };
                    item.Datas["位置"] = colums[3];
                    return item;
                }).ToList();
            return result;
        }
    }
}