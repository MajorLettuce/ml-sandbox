using System.IO;
using MathNet.Numerics.LinearAlgebra;

namespace ML.Model.Transformers
{
    class MnistDataTransformer : DataTransformer
    {
        //public MnistDataTransformer(NetworkModel model) : base(model) { }

        /// <summary>
        /// Transform data file into array of vectors.
        /// </summary>
        /// <param name="file"></param>
        /// <returns></returns>
        public override Matrix<double> TransformData(string file)
        {
            var reader = new BinaryReader(File.Open("train-images.idx3-ubyte", FileMode.Open));

            return Matrix<double>.Build.Dense(1, 2);
        }

        /// <summary>
        /// Transform labels file into a vector.
        /// </summary>
        /// <param name="file"></param>
        /// <returns></returns>
        public override Matrix<double> TransformLabels(string file)
        {
            return Matrix<double>.Build.Dense(1, 2);
        }
    }
}
