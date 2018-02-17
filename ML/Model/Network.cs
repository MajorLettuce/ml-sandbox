using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Data.Text;
using ML.Network;

namespace ML.Model
{
    class Network : NetworkModel
    {
        new NetworkConfig Config;

        /// <summary>
        /// List of network layers.
        /// </summary>
        List<Layer> layers;

        /// <summary>
        /// Artificial neural network model constructor.
        /// </summary>
        /// <returns></returns>
        public Network(string model, NetworkConfig config) : base(model, config)
        {
            Config = config;

            Initialize();
        }

        /// <summary>
        /// Get model info as text string.
        /// </summary>
        /// <returns></returns>
        public override string GetInfo()
        {
            string info = String.Format("inputs: {0}, layers: {1}", Config.Inputs, layers.Count);

            foreach (var layer in layers)
            {
                info += String.Format(
                    "\nlayer: {0}, neurons: {1}, function: {2}",
                    layer.Type,
                    layer.Size,
                    layer.Function.ToString().Split('.').Last().ToLower()
                );
            }

            return info;
        }

        /// <summary>
        /// Initialize current model.
        /// </summary>
        public override void Initialize()
        {
            try
            {
                // Try to initialize model state from files.
                layers = new List<Layer>();

                int count = 0;
                int inputCount = Config.Inputs;
                foreach (var layer in Config.Layers)
                {
                    var state = DelimitedReader.Read<double>(Path(String.Format("layers/{0}.csv", count)));
                    // If either number of neurons or number of inputs
                    // (omitting bias) is wrong, format is incorrect.
                    if (state.RowCount != layer.NeuronCount || state.ColumnCount - 1 != inputCount)
                    {
                        throw new Exception("Incorrect format");
                    }

                    var neurons = new List<Neuron>();
                    foreach (var neuronState in state.ToRowArrays())
                    {
                        var weights = Vector<double>.Build.Dense(neuronState).SubVector(1, neuronState.Length - 1);
                        neurons.Add(new Neuron(inputCount, weights, neuronState[0], layer.Function.ToString()));
                    }
                    layers.Add(new Layer(neurons));

                    // Set current layer neuron count as next layer input count.
                    inputCount = layer.NeuronCount;
                    count++;
                }
            }
            catch
            {
                // If fails, generate new one from scratch.
                layers = new List<Layer>();

                foreach (var layer in Config.Layers)
                {
                    int inputCount;
                    if (layers.Count == 0)
                    {
                        // First layer input count is network input count.
                        inputCount = Config.Inputs;
                    }
                    else
                    {
                        // All subsequent layers input count is previous layer size.
                        inputCount = layers.Last().Size;
                    }
                    layers.Add(Layer.Generate(layer.NeuronCount, inputCount, layer.Function.ToString()));
                }
            }
        }

        /// <summary>
        /// Save currently loaded model.
        /// </summary>
        public override void Save()
        {
            Directory.CreateDirectory(Path("layers"));

            int count = 0;
            foreach (var layer in layers)
            {
                var state = Matrix<double>.Build.Dense(layer.Size, layer.InputCount + 1, (row, column) =>
                {
                    if (column == 0)
                    {
                        // First column is neuron bias.
                        return layer.Neurons[row].Bias;
                    }
                    else
                    {
                        // Everything else is weights.
                        return layer.Neurons[row].Weights.At(column - 1);
                    }
                });
                DelimitedWriter.Write(Path(String.Format("layers/{0}.csv", count)), state);
                count++;
            }
        }

        /// <summary>
        /// Delete currently loaded model.
        /// </summary>
        public override void Delete()
        {
            foreach (var file in Directory.GetFiles(Path("layers")))
            {
                File.Delete(file);
            }
            Directory.Delete(Path("layers"));
        }

        /// <summary>
        /// Process given inputs through the model.
        /// </summary>
        /// <param name="inputs"></param>
        /// <returns></returns>
        public override Vector<double> Process(Vector<double> inputs)
        {
            // Intermediate result vector while going forward through the network.
            Vector<double> intermediate = null;
            foreach (var layer in layers)
            {
                intermediate = layer.Forward(intermediate ?? inputs);
            }
            return intermediate;
        }

        /// <summary>
        /// Run teaching interation.
        /// </summary>
        /// <returns></returns>
        public List<Vector<double>> Teach(Vector<double> targets, Vector<double> outputs)
        {
            // Calculate total squared error vector for each output.
            double error = outputs.Subtract(Process(targets)).PointwisePower(2).Divide(2).Sum();
            // Calculate backpropagation gradients for each weight.
            // This will allow to accumulate them and apply once at the end of the epoch.
            // Each layer has arbitrary amount of weights,
            // so it's not possible to fit them in a matrix, the list is used instead.
            var gradients = new List<Vector<double>>();

            Vector<double> gradient = null;

            for (int i = layers.Count - 1; i >= 0; i--)
            {
                var layer = layers[i];

                // If gradient vector doesn't exist yet, that means this is the last layer.
                if (gradient == null)
                {
                    gradient = layer.Backward(Vector<double>.Build.Dense(layer.Size, 1));
                }
                else
                {
                    gradient = layer.Backward(gradient);
                }

                Console.WriteLine("layer[{0}] gradient: {1}", i, gradient);
                /*
                var deltas = outputs.Subtract(targets).PointwiseMultiply(Vector<double>.Build.Dense(layer.Size, index =>
                {
                    return layer.Function.Derivative(outputs.At(index));
                }));

                Console.WriteLine("deltas: {0}", deltas);
                */
                /*
                var vector = Vector<double>.Build.Dense(layer.Size, index =>
                {
                    return 0;
                });
                gradients.Add(vector);
                */
            }

            return gradients;
        }

        /// <summary>
        /// Run teaching epoch.
        /// </summary>
        /// <returns></returns>
        public override double RunEpoch()
        {
            foreach (var vector in
                Teach(
                    Vector<double>.Build.Dense(new double[] { 0.05, 0.1 }),
                    Vector<double>.Build.Dense(new double[] { 0.01, 0.99 })
                )
            )
            {
                Console.WriteLine(vector);
            }
            return 0;
        }
    }
}
