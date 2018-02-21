using System.IO;
using MathNet.Numerics.LinearAlgebra;

namespace ML.Model.Transformers
{
    class MnistDataTransformer : DataTransformer
    {
        public MnistDataTransformer(NetworkModel model) : base(model) { }

        /// <summary>
        /// Transform data file into array of vectors.
        /// </summary>
        /// <param name="file"></param>
        /// <returns></returns>
        public override Matrix<double> Transform(string file)
        {
            var reader = new BinaryReader(File.OpenRead(file));

            return Matrix<double>.Build.Dense(1, 2);
        }
    }
}
