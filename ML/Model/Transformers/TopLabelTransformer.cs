using System;
using System.Linq;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;

namespace ML.Model.Transformers
{
    class TopLabelTransformer : SingleLabelTransformer
    {
        /// <summary>
        /// Number of maximum labels to list.
        /// </summary>
        protected int count = 5;

        /// <summary>
        /// Threshold to show results only higher than its value.
        /// </summary>
        protected double threshold = 0.01;

        /// <summary>
        /// Label transformer constructor.
        /// </summary>
        /// <param name="model"></param>
        public TopLabelTransformer(NetworkModel model) : base(model) { }

        /// <summary>
        /// Transform output vector into a list of labels.
        /// </summary>
        /// <param name="output"></param>
        /// <param name="file"></param>
        /// <returns></returns>
        public override List<string> TransformOutput(Vector<double> output)
        {
            var labels = LoadLabels();

            var list = new List<string>();

            var dictionary = new Dictionary<string, double>();

            var maxIndex = 0;

            for (int i = 0; i < (count > 0 ? Math.Min(count, output.Count) : output.Count); i++)
            {
                maxIndex = output.MinimumIndex();
                var max = output.Minimum();

                for (int j = 0; j < output.Count; j++)
                {
                    var element = output[j];
                    if (max < element && !dictionary.ContainsValue(element))
                    {
                        max = element;
                        maxIndex = j;
                    }
                }

                if (!dictionary.ContainsValue(max))
                {
                    dictionary.Add(labels[maxIndex], max);
                }
            }

            foreach (var element in dictionary)
            {
                var value = 100 * element.Value / output.Sum();
                if (threshold == 0 || value >= threshold)
                {
                    // Output normalized values which act like probabilities.
                    list.Add(String.Format("{0:f2}% - {1}", value, element.Key));
                }
            }

            return list;
        }
    }
}
