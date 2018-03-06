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

        private Vector<double> weightGradient;
        private Vector<double> inputGradient;
        private Matrix<double> localGradient;

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

            double net = inputs.PointwiseMultiply(Weights).Sum() + Bias;

            // Matrix [number of inputs] x [3 (bias, weights, input vectors)]
            localGradient = Matrix<double>.Build.Dense(InputCount, 3, (row, column) =>
            {
                switch (column)
                {
                    default: // Bias gradient.
                        {
                            // Bias has always input of 1.
                            return Function.Derivative(1);
                        }
                    case 1: // Weights gradient.
                        {
                            // Weight gradient depends on input value.
                            return Function.Derivative(net) * inputs[row];
                        }
                    case 2: // Input gradient.
                        {
                            return Function.Derivative(net) * Weights[row];
                        }
                }
            });

            return Function.Calculate(net);
        }

        /// <summary>
        /// Backpropagate output gradient to the neuron bias, weights and inputs.
        /// </summary>
        /// <param name="gradient">
        /// Output (previous level) gradient scalar (single value/neuron connection).
        /// </param>
        /// <returns></returns>
        public Matrix<double> Backward(double gradient)
        {
            return localGradient.Multiply(gradient);
            // Return gradient vector.
            /*
            return weightGradient.Multiply(gradient);

            return Vector<double>.Build.Dense(InputCount, (index) =>
            {
                //Console.WriteLine("derivative: {0}", Function.Derivative(CachedInputs.At(index)));
                return Function.Derivative(CachedInputs.At(index)) * gradient;
            });
            */
        }
    }
}