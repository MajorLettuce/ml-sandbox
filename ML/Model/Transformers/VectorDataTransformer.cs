using System.IO;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Data.Text;

namespace ML.Model.Transformers
{
    class VectorDataTransformer : IDataTransformer
    {
        /// <summary>
        /// Transform input data into matrix with rows as input vectors.
        /// </summary>
        /// <param name="file"></param>
        /// <returns></returns>
        public Matrix<double> TransformData(string file)
        {
            return DelimitedReader.Read<double>(file);
        }

        /// <summary>
        /// Transform labels file into a vector.
        /// </summary>
        /// <param name="file"></param>
        /// <returns></returns>
        public Matrix<double> TransformLabels(string file)
        {
            var labels = File.ReadAllLines(file);

            var vectors = new List<Vector<double>>();

            for (int i = 0; i < labels.Length; i++)
            {
                vectors.Add(
                    Vector<double>.Build.Dense(labels.Length, index =>
                    {
                        return index == i ? 1 : 0;
                    })
                );
            }

            return Matrix<double>.Build.DenseOfRowVectors(vectors);
        }
    }
}
