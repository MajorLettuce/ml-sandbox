using MathNet.Numerics.LinearAlgebra;

namespace ML.Model.Transformers
{
    interface IDataTransformer
    {
        /// <summary>
        /// Transform data file into array of vectors.
        /// </summary>
        /// <param name="file"></param>
        /// <returns></returns>
        Matrix<double> TransformData(string file);

        /// <summary>
        /// Transform labels file into array of vectors.
        /// </summary>
        /// <param name="file"></param>
        /// <returns></returns>
        Matrix<double> TransformLabels(string file);
    }
}
