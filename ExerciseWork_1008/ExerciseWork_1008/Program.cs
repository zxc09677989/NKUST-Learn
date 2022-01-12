using System;
using ExerciseWork_1008.Unit;
using System.Linq;
namespace ExerciseWork_1008
{
    class Program
    {
        static void Main(string[] args)
        {
            //取得路徑
            string FullFileName = Filepath.GetFullepath("高雄市108年路口監視器地點.csv");

            var csvService = new ExerciseWork_1008.sencices.ImportcsvService();
            var csvData = csvService.LoadFormfile(FullFileName);
            //
            Console.WriteLine(string.Format("分析完成共有{0}筆資料", csvData.Count));
            var groupDatas = csvData.GroupBy(x => x.派出所 ,y => y).ToList();
            groupDatas.ForEach(x =>
            {
                var items = x.ToList();
                Console.WriteLine($"派出所:{x.Key}數量:{x.ToList().Count}");
                items.ForEach(x =>
                {
                    Console.WriteLine($"分局:{x.分局} 派出所:{x.派出所} 編號:{x.警編號}");
                });
            });
           // csvData.ForEach(x =>
           // {
           //    Console.WriteLine($"分局:{x.分局} 派出所:{x.派出所} 編號:{x.警編號}");
           // });
            Console.ReadKey();
        }
    }
}
