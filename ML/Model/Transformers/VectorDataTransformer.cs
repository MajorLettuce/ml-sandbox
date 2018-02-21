using System.IO;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Data.Text;

namespace ML.Model.Transformers
{
    class VectorDataTransformer : DataTransformer
    {
        public VectorDataTransformer(NetworkModel model) : base(model) { }

        /// <summary>
        /// Transform input data into matrix with rows as input vectors.
        /// </summary>
        /// <param name="file"></param>
        /// <returns></returns>
        public override Matrix<double> Transform(string file)
        {
            return DelimitedReader.Read<double>(file);
        }
    }
}
