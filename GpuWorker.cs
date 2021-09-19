using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Microsoft.ML.OnnxRuntime;

namespace gvaduha
{
    static class ActionDiagExt
    {
        static public long GetRunTime(this Action action)
        {
            var sw = new Stopwatch();
            sw.Start();

            action();

            sw.Stop();
            return sw.ElapsedMilliseconds;
        }
    }

    public class TestResult
    {
        public string Name;
        public long[] RunTime;
    }

    class GpuWorker
    {
        int _gpu;
        int _batchSize;
        InDataProvider _dp;
        InferenceSession _session;
        string _inputName;

        public GpuWorker(byte[] model, int gpu, InDataProvider dp, int batchSize)
        {
            _gpu = gpu;
            _dp = dp;
            _batchSize = batchSize;

            _session = new InferenceSession(model, SessionOptions.MakeSessionOptionWithCudaProvider(gpu));
            _inputName = _session.InputMetadata.Keys.First();

            //warmup
            var (w, h, c, d) = _dp.GetData();
            _session.Run(new List<NamedOnnxValue> { d.CreateTensor((w, h, c), _inputName) });

            {
                //HACK: How to get friendly name of device?
                //var x = new OrtApi();
                //var ov = OrtValue.CreateFromTensorObject()
            }
        }

        public TestResult Run(int runCount)
        {
            var samples = new long[runCount];

            for (int i = 0; i < runCount; ++i)
                samples[i] = ((Action) RunOnce).GetRunTime();

            return new TestResult { Name = _gpu.ToString(), RunTime = samples };
        }

        public void RunOnce()
        {
            try
            {
                (int w, int h, int c, byte[] d)[] imgs = _dp.GetData(_batchSize).ToArray();

                var data = imgs.Select(x => x.d).ToArray();

                var t = data.CreateTensor((imgs[0].w, imgs[0].h, imgs[0].c), _inputName)
                            .ToArray();

                _session.Run(t);
            }
            catch (Exception e)
            {
                throw new ApplicationException($"worker@{_gpu} failed", e);
            }
        }
    }
}
