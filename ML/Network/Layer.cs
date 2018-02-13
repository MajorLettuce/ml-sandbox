using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using ML.Network.ActivationFunction;

namespace ML.Network
{
    class Layer
    {
        /// <summary>
        /// List of neurons in the layer.
        /// </summary>
        List<Neuron> neurons;

        /// <summary>
        /// Number of neurons in layer.
        /// </summary>
        public int Size { protected set; get; }

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
        public string Function { protected set; get; }

        /// <summary>
        /// Neuron bias.
        /// </summary>
        public double Bias { protected set; get; }

        /// <summary>
        /// Neuron network layer constructor.
        /// </summary>
        /// <param name="size"></param>
        /// <param name="inputs"></param>
        /// <param name="function"></param>
        /// <param name="bias"></param>
        /// <param name="weights"></param>
        public Layer(int size, int inputs, Vector<double> weights, double bias, string function)
        {
            neurons = new List<Neuron>();

            for (int i = 0; i < size; i++)
            {
                var neuron = new Neuron(inputs, weights, bias, function);
                neurons.Add(neuron);
            }
        }

        /// <summary>
        /// Process given inputs through the layer neurons.
        /// </summary>
        /// <param name="inputs"></param>
        /// <returns></returns>
        public Vector<double> Process(Vector<double> inputs)
        {
            if (Inputs != inputs.Count)
            {
                throw new Exception("Incorrect number of inputs.");
            }

            return Vector<double>.Build.Dense(Size, index =>
            {
                return neurons[index].Process(inputs);
            });
        }
    }
}
