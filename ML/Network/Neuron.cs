using System;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;
using ML.Network.ActivationFunction;

namespace ML.Network
{
    class Neuron
    {
        /// <summary>
        /// Number of neuron inputs.
        /// </summary>
        public int Inputs { protected set; get; }

        /// <summary>
        /// Vector of input weights.
        /// </summary>
        public Vector<double> Weights { protected set; get; }

        /// <summary>
        /// Neuron result activation function.
        /// </summary>
        public IActivationFunction Function { protected set; get; }

        /// <summary>
        /// Neuron constructor.
        /// </summary>
        /// <param name="inputs"></param>
        /// <param name="weights"></param>
        /// <param name="bias"></param>
        /// <param name="function"></param>
        public Neuron(int inputs, Vector<double> weights, double bias, string function)
        {
            if (weights.Count != inputs)
            {
                throw new Exception("Number of weights must be equal to the number of inputs.");
            }

            Inputs = inputs;
            Weights = weights;
            Bias = bias;

            var functionType = Type.GetType(String.Format("ML.Network.ActivationFunction.{0}", function));

            try
            {
                Function = Activator.CreateInstance(functionType) as IActivationFunction;
            }
            catch (Exception e)
            {
                throw new Exception("Unable to load activation function '" + function + "'.", e);
            }
        }

        /// <summary>
        /// Neuron bias.
        /// </summary>
        public double Bias { set; get; }

        /// <summary>
        /// Generate new neuron with random parameters.
        /// </summary>
        /// <param name="inputs"></param>
        /// <param name="function"></param>
        /// <returns></returns>
        public static Neuron Generate(int inputs, string function)
        {
            var weights = Vector<double>.Build.Random(inputs, new ContinuousUniform(-1, 1));
            var bias = new Random().NextDouble() * 2 - 1;

            return new Neuron(inputs, weights, bias, function);
        }

        /// <summary>
        /// Process given inputs through the neuron.
        /// </summary>
        /// <param name="inputs"></param>
        /// <returns></returns>
        public double Process(Vector<double> inputs)
        {
            if (Inputs != inputs.Count)
            {
                throw new Exception("Incorrect number of inputs.");
            }

            double result = 0;

            for (int i = 0; i < inputs.Count; i++)
            {
                result += inputs[i] * Weights[i] + Bias;
            }

            return Function.Calculate(result);
        }
    }
}
