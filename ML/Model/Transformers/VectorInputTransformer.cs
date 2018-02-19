using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Data.Text;

namespace ML.Model.Transformers
{
    class VectorInputTransformer : IInputTransformer
    {
        /// <summary>
        /// Transform data file into array of vectors.
        /// </summary>
        /// <param name="file"></param>
        /// <returns></returns>
        public Matrix<double> Transform(string file)
        {
            return DelimitedReader.Read<double>(file);
        }
    }
}
