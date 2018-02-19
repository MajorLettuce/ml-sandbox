using MathNet.Numerics.LinearAlgebra;

namespace ML.Model.Transformers
{
    interface IOutputTransformer
    {
        /// <summary>
        /// Transform output vector to string.
        /// </summary>
        /// <param name="output"></param>
        /// <returns></returns>
        string Transform(Vector<double> output);
    }
}
