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
        /// Load training labels.
        /// </summary>
        /// <returns></returns>
        protected override string[] LoadTrainLabels()
        {
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

            return trainLabels;
        }
    }
}
