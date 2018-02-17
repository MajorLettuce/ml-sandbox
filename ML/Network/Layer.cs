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
        public List<Neuron> Neurons { protected set; get; }

        /// <summary>
        /// Type of the layer.
        /// </summary>
        public string Type
        {
            get
            {
                return "fc";
            }
        }

        /// <summary>
        /// Number of neurons in layer.
        /// </summary>
        public int Size
        {
            get
            {
                return Neurons.Count;
            }
        }

        /// <summary>
        /// Number of neuron inputs.
        /// </summary>
        public int InputCount
        {
            get
            {
                return Neurons[0].InputCount;
            }
        }

        /// <summary>
        /// Layer function.
        /// </summary>
        public IActivationFunction Function
        {
            get
            {
                return Neurons[0].Function;
            }
        }

        /// <summary>
        /// Neuron network layer constructor.
        /// </summary>
        /// <param name="neurons"></param>
        public Layer(List<Neuron> neurons)
        {
            if (neurons.Count == 0)
            {
                throw new Exception("Layer must have at least one neuron.");
            }

            Neurons = new List<Neuron>();

            for (int i = 0; i < neurons.Count; i++)
            {
                Neurons.Add(neurons[i]);
            }
        }

        /// <summary>
        /// Generate new layer with randomly generated neurons.
        /// </summary>
        /// <param name="size"></param>
        /// <param name="inputCount"></param>
        /// <param name="function"></param>
        /// <returns></returns>
        public static Layer Generate(int size, int inputCount, string function)
        {
            var neurons = new List<Neuron>();

            for (int i = 0; i < size; i++)
            {
                neurons.Add(Neuron.Generate(inputCount, function));
            }

            return new Layer(neurons);
        }

        /// <summary>
        /// Process given inputs through the layer neurons.
        /// </summary>
        /// <param name="inputs"></param>
        /// <returns></returns>
        public Vector<double> Forward(Vector<double> inputs)
        {
            if (InputCount != inputs.Count)
            {
                throw new Exception("Incorrect number of inputs.");
            }

            return Vector<double>.Build.Dense(Size, index =>
            {
                return Neurons[index].Forward(inputs);
            });
        }
    }
}
