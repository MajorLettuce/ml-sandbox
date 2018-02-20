using System.IO;
using System.Linq;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Data.Text;

namespace ML.Model.Transformers
{
    class SingleLabelTransformer : LabelTransformer
    {
        protected string[] cachedLabels;

        protected string[] cachedTrainLabels;

        protected Matrix<double> cachedTrainLabelsMatrix;

        /// <summary>
        /// Label transformer constructor.
        /// </summary>
        /// <param name="model"></param>
        public SingleLabelTransformer(NetworkModel model) : base(model) { }

        protected string[] LoadLabels()
        {
            var labels = cachedLabels;

            if (labels == null)
            {
                labels = File.ReadAllLines(model.Path("labels"));
                cachedLabels = labels;
            }

            return labels;
        }

        /// <summary>
        /// Transform output vector into a list of labels.
        /// </summary>
        /// <param name="output"></param>
        /// <param name="file"></param>
        /// <returns></returns>
        public override List<string> TransformOutput(Vector<double> output)
        {
            var labels = LoadLabels();

            return new List<string> { labels[output.MaximumIndex()] };
        }

        /// <summary>
        /// Transform train labels into a matrix with rows as target values
        /// by using original label list.
        /// </summary>
        /// <param name="file"></param>
        /// <returns></returns>
        public override Matrix<double> TransformLabels()
        {
            if (cachedTrainLabelsMatrix != null)
            {
                return cachedTrainLabelsMatrix;
            }

            var trainLabels = cachedTrainLabels;

            if (trainLabels == null)
            {
                trainLabels = File.ReadAllLines(model.Path(model.Config.TrainLabels));
                cachedTrainLabels = trainLabels;
            }

            var realLabels = LoadLabels().ToList();

            var diagonal = Matrix<double>.Build.DenseDiagonal(realLabels.Count, 1);

            var matrix = Matrix<double>.Build.Dense(trainLabels.Length, realLabels.Count);

            System.Diagnostics.Debug.WriteLine(diagonal);

            for (int i = 0; i < matrix.RowCount; i++)
            {
                matrix.SetRow(i, diagonal.Row(realLabels.FindIndex(label => label == trainLabels[i])));
            }

            System.Diagnostics.Debug.WriteLine(matrix);

            cachedTrainLabelsMatrix = matrix;

            return matrix;
        }
    }
}
