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

            cachedData = Matrix<double>.Build.Dense(count, rows * columns, (row, column) =>
            {
                var pixel = reader.ReadByte();
                if (pixel == 0)
                {
                    return 0;
                }
                else
                {
                    return pixel / 255.0;
                }
            });
            /*
            System.Diagnostics.Debug.WriteLine(cachedData);

            System.Console.ReadLine();
            */
            return cachedData;
        }
    }
}
