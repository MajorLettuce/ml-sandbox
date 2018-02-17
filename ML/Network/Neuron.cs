using System;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using ML.Network.ActivationFunction;

namespace ML.Network
{
    class Neuron
    {
        /// <summary>
        /// Number of neuron inputs.
        /// </summary>
        public int InputCount { protected set; get; }

        /// <summary>
        /// Vector of input weights.
        /// </summary>
        public Vector<double> Weights { protected set; get; }

        /// <summary>
        /// Neuron bias.
        /// </summary>
        public double Bias { set; get; }

        /// <summary>
        /// Neuron result activation function.
        /// </summary>
        public IActivationFunction Function { protected set; get; }

        /// <summary>
        /// Vector of cached intermediate inputs used for backpropagation.
        /// </summary>
        protected Vector<double> CachedInputs { get; set; }

        /// <summary>
        /// Neuron constructor.
        /// </summary>
        /// <param name="inputCount"></param>
        /// <param name="weights"></param>
        /// <param name="bias"></param>
        /// <param name="function"></param>
        public Neuron(int inputCount, Vector<double> weights, double bias, string function)
        {
            if (weights.Count != inputCount)
            {
                throw new Exception("Number of weights must be equal to the number of inputs.");
            }

            InputCount = inputCount;
            Weights = weights;
            Bias = bias;

            try
            {
                Function = Activator.CreateInstance(
                    Type.GetType(String.Format("ML.Network.ActivationFunction.{0}", function))
                ) as IActivationFunction;
            }
            catch (Exception e)
            {
                throw new Exception("Unable to load activation function '" + function + "'.", e);
            }
        }

        /// <summary>
        /// Generate new neuron with random parameters.
        /// </summary>
        /// <param name="inputCount"></param>
        /// <param name="function"></param>
        /// <returns></returns>
        public static Neuron Generate(int inputCount, string function)
        {
            var weights = Vector<double>.Build.Random(inputCount, new ContinuousUniform(-1, 1));
            var bias = new ContinuousUniform(-1, 1).RandomSource.NextDouble();

            return new Neuron(inputCount, weights, bias, function);
        }

        /// <summary>
        /// Process given inputs through the neuron.
        /// </summary>
        /// <param name="inputs"></param>
        /// <returns></returns>
        public double Forward(Vector<double> inputs)
        {
            if (InputCount != inputs.Count)
            {
                throw new Exception("Incorrect number of inputs.");
            }

            CachedInputs = inputs.Clone();

            double result = 0;

            for (int i = 0; i < inputs.Count; i++)
            {
                result += inputs[i] * Weights[i];
            }

            result += Bias;

            return Function.Calculate(result);
        }

        /// <summary>
        /// Backpropagate output gradient to the neuron inputs.
        /// </summary>
        /// <param name="gradient">
        /// Output (previous level) gradient scalar (single value/neuron connection).
        /// </param>
        /// <returns></returns>
        public Vector<double> Backward(double gradient)
        {
            Console.WriteLine("gradient: {0}", gradient);
            // Return gradient vector.
            return Vector<double>.Build.Dense(InputCount, (index) =>
            {
                Console.WriteLine("derivative: {0}", Function.Derivative(CachedInputs.At(index)));
                return Function.Derivative(CachedInputs.At(index)) * gradient;
            });
        }
    }
}
