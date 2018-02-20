using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Data.Text;
using ML.Utility;

namespace ML.Model.Transformers
{
    class MnistLabelTransformer : SingleLabelTransformer
    {
        /// <summary>
        /// Label transformer constructor.
        /// </summary>
        /// <param name="model"></param>
        public MnistLabelTransformer(NetworkModel model) : base(model) { }

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
                var reader = new BigEndianBinaryReader(File.OpenRead(model.Path(model.Config.Train.Labels)));

                if (reader.ReadInt32() != 2049)
                {
                    throw new Exception("Invalid train label file format.");
                }

                var count = reader.ReadInt32();

                trainLabels = new string[count];

                for (int i = 0; i < count; i++)
                {
                    trainLabels[i] = reader.ReadByte().ToString();
                }

                cachedTrainLabels = trainLabels;
            }

            var realLabels = LoadLabels().ToList();

            var diagonal = Matrix<double>.Build.DenseDiagonal(realLabels.Count, 1);

            var matrix = Matrix<double>.Build.Dense(trainLabels.Length, realLabels.Count);

            for (int i = 0; i < matrix.RowCount; i++)
            {
                matrix.SetRow(i, diagonal.Row(realLabels.FindIndex(label => label == trainLabels[i])));
            }

            cachedTrainLabelsMatrix = matrix;

            return matrix;
        }
    }
}
