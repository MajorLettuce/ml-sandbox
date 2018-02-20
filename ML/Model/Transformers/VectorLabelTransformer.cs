using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Data.Text;

namespace ML.Model.Transformers
{
    class VectorLabelTransformer : ILabelTransformer
    {
        /// <summary>
        /// Transform output vector into a list of labels.
        /// </summary>
        /// <param name="output"></param>
        /// <returns></returns>
        public List<string> TransformOutput(Vector<double> output)
        {
            return new List<string> { output.ToVectorString() };
        }

        /// <summary>
        /// Transform labels into a matrix with rows as target values.
        /// </summary>
        /// <param name="file"></param>
        /// <returns></returns>
        public Matrix<double> TransformLabels(string file)
        {
            return DelimitedReader.Read<double>(file);
        }
    }
}
