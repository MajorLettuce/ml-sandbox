using MathNet.Numerics.LinearAlgebra;

namespace ML.Model.Transformers
{
    interface IInputTransformer
    {
        /// <summary>
        /// Transform data file into array of vectors.
        /// </summary>
        /// <param name="file"></param>
        /// <returns></returns>
        Matrix<double> Transform(string file);
    }
}
