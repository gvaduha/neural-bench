using System;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.CommandLineUtils;

namespace gvaduha
{
    class Program
    {
        static async Task<int> Main(string[] args)
        {
            var cla = new CommandLineApplication(throwOnUnexpectedArg: false);
            var amodel = cla.Option("-m", "file name of onnx model with one byte array (image) input, REQUIRED", CommandOptionType.SingleValue);
            var aruncnt = cla.Option("-n", "times to run test, default 1000", CommandOptionType.SingleValue);
            var agpulist = cla.Option("-g", "id of gpu devices separated by ';', default 0", CommandOptionType.SingleValue);
            var abtchsize = cla.Option("-b", "batch size, default is 1", CommandOptionType.SingleValue);
            var aimgsize = cla.Option("-s", "input image size in WIDTHxHEIGHT, default 800x600", CommandOptionType.SingleValue);

            byte[] model;
            int runcnt;
            int[] gpus;
            int btchsize;
            Size imgsize;

            try
            {
                cla.Execute(args);

                
                model = File.ReadAllBytes(amodel.Value());
                runcnt = aruncnt.HasValue() ? Convert.ToInt32(aruncnt.Value()) : 1000;
                gpus = agpulist.HasValue() ? agpulist.Value().Split(';').Select(x => Convert.ToInt32(x)).ToArray() : new int[] {0};
                btchsize = abtchsize.HasValue() ? Convert.ToInt32(abtchsize.Value()) : 1;
                imgsize = aimgsize.HasValue()
                    ? new Func<string, Size>((s) => { 
                            var dims = s.Split('x').Select(x => Convert.ToInt32(x)).ToArray();
                            return new Size(dims[0], dims[1]);
                        })(aimgsize.Value())
                    : new Size(800, 600);
            }
            catch (Exception e)
            {
                cla.ShowHelp();
                Console.WriteLine($"Error: {e.Message}");
                return -1;
            }

            var bt = new BenchTest(model, runcnt, gpus, btchsize, imgsize);

            var results = await bt.Run();

            Console.WriteLine($"Completed {runcnt} runs with batchsize:{btchsize}, imgsize:{imgsize.Width}x{imgsize.Height}:");
            foreach (var res in results)
                Console.WriteLine($"{res.Name}: avg: {res.RunTime.Average()}, min: {res.RunTime.Min()}, max: {res.RunTime.Max()}");

            return 0;
        }
    }

    class BenchTest
    {
        int _runCount;
        InDataProvider _dp;
        GpuWorker[] _workers;

        public BenchTest(byte[] model, int runCount, int[] gpus, int batchSize, Size imgSize)
        {
            _runCount = runCount;
            _dp = new InDataProvider(imgSize);
            _workers = new GpuWorker[gpus.Length];

            for (var i = 0; i < gpus.Length; ++i)
                _workers[i] = new GpuWorker(model, gpus[i], _dp, batchSize);
        }

        public async Task<TestResult[]> Run()
        {
            var gpuTasks = new Task<TestResult>[_workers.Length];

            for (var i = 0; i < _workers.Length; ++i)
            {
                var i2capt = i;
                var t = new Task<TestResult>(() => _workers[i2capt].Run(_runCount), TaskCreationOptions.LongRunning);
                gpuTasks[i] = t;
                t.Start();
            }

            var results = await Task.WhenAll(gpuTasks);

            return results;
        }
    }
}
