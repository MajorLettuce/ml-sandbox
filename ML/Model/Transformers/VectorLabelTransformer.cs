using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Data.Text;

namespace ML.Model.Transformers
{
    class VectorLabelTransformer : LabelTransformer
    {
        /// <summary>
        /// Label transformer constructor.
        /// </summary>
        /// <param name="model"></param>
        public VectorLabelTransformer(NetworkModel model) : base(model) { }

        /// <summary>
        /// Transform output vector into a list of labels.
        /// </summary>
        /// <param name="output"></param>
        /// <returns></returns>
        public override List<string> TransformOutput(Vector<double> output)
        {
            return new List<string> { output.ToVectorString() };
        }

        /// <summary>
        /// Transform labels into a matrix with rows as target values.
        /// </summary>
        /// <param name="file"></param>
        /// <returns></returns>
        public override Matrix<double> TransformLabels()
        {
            return DelimitedReader.Read<double>(model.Path(model.Config.Train.Labels));
        }
    }
}
