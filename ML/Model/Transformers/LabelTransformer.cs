using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;

namespace ML.Model.Transformers
{
    abstract class LabelTransformer
    {
        /// <summary>
        /// Model this transformer belongs to.
        /// </summary>
        protected NetworkModel model;

        /// <summary>
        /// Label transformer constructor.
        /// </summary>
        /// <param name="model"></param>
        public LabelTransformer(NetworkModel model)
        {
            this.model = model;
        }

        /// <summary>
        /// Transform output vector into a list of labels.
        /// </summary>
        /// <param name="output"></param>
        /// <returns></returns>
        public abstract List<string> TransformOutput(Vector<double> output);

        /// <summary>
        /// Transform labels into a matrix with rows as target values.
        /// </summary>
        /// <param name="file"></param>
        /// <returns></returns>
        public abstract Matrix<double> TransformLabels();
    }
}
