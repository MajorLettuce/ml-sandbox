using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;

namespace ML.Model.Transformers
{
    interface ILabelTransformer
    {
        /// <summary>
        /// Transform output vector into a list of labels.
        /// </summary>
        /// <param name="output"></param>
        /// <returns></returns>
        List<string> TransformOutput(Vector<double> output);

        /// <summary>
        /// Transform labels into a matrix with rows as target values.
        /// </summary>
        /// <param name="file"></param>
        /// <returns></returns>
        Matrix<double> TransformLabels(string file);
    }
}
