using System;
using System.Collections.Generic;
using System.Drawing;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace gvaduha
{
    class InDataProvider
    {
        Size _size;
        readonly byte[] _etalon;
        const int Channels = 3;
        int _datalen;

        public unsafe InDataProvider(Size size, byte[] data)
        {
            _size = size;
            _datalen = size.Width * size.Height * Channels;
            _etalon = new byte[_datalen];

            fixed (byte* src = data)
            fixed (byte* dst = _etalon)
            {
                Buffer.MemoryCopy(src, dst, _datalen, _datalen);
            }
        }

        public InDataProvider(Size size)
            : this(size, CreateRandomArray(size))
        { }

        static byte[] CreateRandomArray(Size size)
        {
            var rnd = new Random(DateTime.Now.Millisecond);
            var datalen = size.Width * size.Height * Channels;
            byte[] data = new byte[datalen];
            rnd.NextBytes(data);
            return data;
        }

        public unsafe (int w, int h, int c, byte[] data) GetData()
        {
            var data = new byte[_datalen];
            fixed (byte* src = _etalon)
            fixed (byte* dst = data)
            {
                Buffer.MemoryCopy(src, dst, _datalen, _datalen);
            }

            return (_size.Width, _size.Height, Channels, data);
        }

        public IEnumerable<(int w, int h, int c, byte[] data)> GetData(int n)
        {
            for (;n>0;--n)
                yield return GetData();
        }
    }

    static class TensorMakeHelper
    {
        public static NamedOnnxValue CreateTensor(this byte[] data, (int w, int h, int c) dim, string tensorName)
        {
            Tensor<byte> t = new DenseTensor<byte>(new Memory<byte>(data), new[] { dim.h, dim.w, dim.c });

            return NamedOnnxValue.CreateFromTensor(tensorName, t);
        }

        public static IEnumerable<NamedOnnxValue> CreateTensor(this byte[][] data, (int w, int h, int c) dim, string tensorName)
        {
            foreach (var datum in data)
                yield return datum.CreateTensor(dim, tensorName);
        }
    }
}
