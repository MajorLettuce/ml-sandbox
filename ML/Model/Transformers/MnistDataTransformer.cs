using System;
using System.IO;
using MathNet.Numerics.LinearAlgebra;
using ML.Utility;

namespace ML.Model.Transformers
{
    class MnistDataTransformer : DataTransformer
    {
        public MnistDataTransformer(NetworkModel model) : base(model) { }


        protected Matrix<double> cachedData;

        /// <summary>
        /// Transform data file into array of vectors.
        /// </summary>
        /// <param name="file"></param>
        /// <returns></returns>
        public override Matrix<double> Transform(string file)
        {
            if (cachedData != null)
            {
                return cachedData;
            }

            var reader = new BigEndianBinaryReader(File.OpenRead(file));

            if (reader.ReadInt32() != 2051)
            {
                throw new Exception("Invalid data file format.");
            }

            var count = reader.ReadInt32();
            var rows = reader.ReadInt32();
            var columns = reader.ReadInt32();

            var vectors = new Vector<double>[count];

            for (int i = 0; i < count; i++)
            {
                vectors[i] = Vector<double>.Build.Dense(rows * columns, index =>
                {
                    return reader.ReadByte() / 255.0;
                });
            }

            cachedData = Matrix<double>.Build.DenseOfRowVectors(vectors);
            /*
            foreach (var image in Directory.EnumerateFiles(model.Path("dataset")))
            {
                File.Delete(image);
            }

            Directory.CreateDirectory(model.Path("dataset"));

            foreach (var image in cachedData.EnumerateRowsIndexed())
            {
                var bitmap = new Bitmap(28, 28);

                for (int i = 0; i < image.Item2.Count; i++)
                {
                    var color = (byte)(image.Item2.At(i) * Byte.MaxValue);
                    bitmap.SetPixel(i % 28, i / 28, Color.FromArgb(color, color, color));
                }
                bitmap.Save(String.Format(model.Path("dataset/{0}.png"), image.Item1 + 1), System.Drawing.Imaging.ImageFormat.Png);
                bitmap.Dispose();
            }
            */
            /*
            var v = cachedData.Row(0);
            for (int i = 0; i < v.Count; i++)
            {
                if (i % 28 == 0)
                {
                    System.Diagnostics.Debug.Write(String.Format("\n{0:d3}: ", i));
                }
                System.Diagnostics.Debug.Write(String.Format(" {0:f3}", v[i]));
            }
            */
            /*
            System.Diagnostics.Debug.WriteLine(cachedData.Row(0));

            System.Console.ReadLine();
            */
            return cachedData;
        }
    }
}
