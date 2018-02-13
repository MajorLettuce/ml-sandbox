using System.IO;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Data.Text;
using ML.Network;
using System;

namespace ML.Model
{
    class Perceptron : NetworkModel
    {
        Neuron perceptron;

        new PerceptronConfig Config;

        string stateFile;

        string sampleFile;

        /// <summary>
        /// Perceptron model constructor.
        /// </summary>
        /// <param name="model"></param>
        /// <param name="config"></param>
        public Perceptron(string model, PerceptronConfig config) : base(model, config)
        {
            Config = config;

            stateFile = "state.csv";

            sampleFile = "samples.csv";

            try
            {
                var state = DelimitedReader.Read<double>(Path(stateFile)).Row(0);
                var bias = state[0];
                var weights = state.SubVector(1, state.Count - 1);
                perceptron = new Neuron(config.Inputs, weights, bias, Config.Function.ToString());
            }
            catch
            {
                perceptron = Generate();
            }

        }

        /// <summary>
        /// Generate new perceptron with random parameters.
        /// </summary>
        /// <returns></returns>
        protected Neuron Generate()
        {
            return Neuron.Generate(Config.Inputs, Config.Function.ToString());
        }

        /// <summary>
        /// Save currently loaded model.
        /// </summary>
        override public void Save()
        {
            var state = Vector<double>.Build.Dense(perceptron.Weights.Count + 1, (index) =>
            {
                if (index == 0)
                {
                    return perceptron.Bias;
                }
                else
                {
                    return perceptron.Weights.At(index - 1);
                }
            });

            DelimitedWriter.Write<double>(Path(stateFile), state.ToRowMatrix());
        }

        /// <summary>
        /// Delete currently loaded model.
        /// </summary>
        override public void Delete()
        {
            var file = Path(stateFile);

            if (File.Exists(file))
            {
                File.Delete(file);
            }
        }

        /// <summary>
        /// Get model info as text string.
        /// </summary>
        /// <returns></returns>
        override public string GetInfo()
        {
            string result = "";

            result += "Weights:\n";
            result += perceptron.Weights.ToVectorString();
            result += "\n";
            result += "Bias: " + perceptron.Bias;

            return result;
        }

        /// <summary>
        /// Process given inputs through the model.
        /// </summary>
        /// <param name="inputs"></param>
        /// <returns></returns>
        override public Vector<double> Process(Vector<double> inputs)
        {
            return Vector<double>.Build.DenseOfArray(new double[] { perceptron.Process(inputs) });
        }

        /// <summary>
        /// Run teaching interation.
        /// </summary>
        override public void Teach(Vector<double> inputs, Vector<double> outputs)
        {
            if (perceptron.Inputs != inputs.Count)
            {
                throw new Exception("Incorrect number of inputs.");
            }

            var error = Process(inputs).Subtract(outputs).At(0);

            for (var i = 0; i < perceptron.Weights.Count; i++)
            {
                var weight = perceptron.Weights.At(i);
                weight -= Config.LearningRate * error * inputs[i];
                perceptron.Weights.At(i, weight);
            }
            // Delta sign is reversed because e = (y - t) instead of (t - y)
            perceptron.Bias -= Config.LearningRate * error;
        }

        /// <summary>
        /// Run teaching epoch using online teaching method.
        /// Weight updates are done on for each sample individually.
        /// </summary>
        override public void RunEpoch()
        {
            var samples = DelimitedReader.Read<double>(Path(sampleFile));
            var permutation = Combinatorics.GeneratePermutation(samples.RowCount);
            foreach (var index in permutation)
            {
                var inputs = samples.Row(index).SubVector(0, perceptron.Inputs);
                var outputs = samples.Row(index).SubVector(perceptron.Inputs, 1);
                Teach(inputs, outputs);
            }
        }
    }
}
