using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Data.Text;

namespace ML.Model.Transformers
{
    class VectorOutputTransformer : IOutputTransformer
    {
        /// <summary>
        /// Transform output vector to string.
        /// </summary>
        /// <param name="output"></param>
        /// <returns></returns>
        public string Transform(Vector<double> output)
        {
            return output.ToString();
        }
    }
}
