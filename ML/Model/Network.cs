using System;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Collections.Generic;
using MathNet.Numerics;
using MathNet.Numerics.Data.Text;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;
using ML.Network;
using ML.Model.Transformers;

namespace ML.Model
{
    class Network : NetworkModel
    {
        new NetworkConfig Config;

        /// <summary>
        /// List of network layers.
        /// </summary>
        List<Layer> layers;

        protected Matrix<double>[] accumulatedGradient;

        protected double accumulatedCost;

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
                    var state = DelimitedReader.Read<double>(Path(String.Format("layers/{0}", count)));
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
                Loaded = true;
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
                DelimitedWriter.Write(Path(String.Format("layers/{0}", count)), state);
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
        /// Show example and provide target output to compare current network result to.
        /// </summary>
        /// <param name="inputs"></param>
        /// <param name="targets"></param>
        /// <param name="cost"></param>
        /// <param name="gradients"></param>
        /// <returns></returns>
        public void Train(Vector<double> inputs, Vector<double> targets, out double cost, out List<Matrix<double>> gradients)
        {
            // Calculate total squared error vector for each output.
            //double error = targets.Subtract(Process(inputs)).PointwisePower(2).Divide(2).Sum();
            //Console.WriteLine("error {0}", error);
            // Cost of current training example.
            cost = Process(inputs).Subtract(targets).PointwisePower(2).Divide(2).Sum();
            // Calculate backpropagation gradients for each weight.
            // This will allow to accumulate them and apply once at the end of the epoch.
            // Each layer has arbitrary amount of weights,
            // so it's not possible to fit them in a matrix, the list is used instead.
            gradients = new List<Matrix<double>>();

            Matrix<double> gradient = null;

            // Go through each layer of the network in opposite direction.
            for (int i = layers.Count - 1; i >= 0; i--)
            {
                Layer previousLayer = null;
                if (i < layers.Count - 1)
                {
                    previousLayer = layers[i + 1];
                }
                Layer currentLayer = layers[i];

                // If gradient vector doesn't exist yet, that means this is the last layer.
                if (gradient == null)
                {
                    // Feed test inputs to the network to update neuron's
                    // local gradients and get error gradient to use as starting gradient.
                    // TODO: Change to calculate arbitrary error gradient, not just squared error.
                    // E = sum(t - out)
                    var diff = targets.Subtract(Process(inputs));

                    gradient = currentLayer.Backward(diff);
                    //var dEdOut = currentLayer.Intermediate.Subtract(targets);

                    //var dOutdNetOut = layer.Forward()

                    //gradient = layer.Backward(Vector<double>.Build.Dense(layer.Size, 1));
                }
                else
                {
                    var diff = Vector<double>.Build.Dense(currentLayer.Size);

                    // Take previous layer rows, and sum all output gradient columns (3),
                    // corresponding to each neuron.
                    // Number of rows in a group is number of neurons in current layer.
                    for (int j = 0; j < gradient.RowCount; j += previousLayer.Size)
                    {
                        double outputGradientSum = 0;
                        for (int k = 0; k < previousLayer.Size; k++)
                        {
                            outputGradientSum += gradient.Row(j + k).At(2);
                        }
                        diff.At(j / previousLayer.Size, outputGradientSum);
                    }

                    gradient = currentLayer.Backward(diff);
                }

                var matrix = Matrix<double>.Build.Dense(currentLayer.Size, currentLayer.InputCount + 1);

                for (int j = 0; j < currentLayer.Size; j++)
                {
                    var neuronBackward = gradient.SubMatrix(
                        j * currentLayer.InputCount,
                        currentLayer.InputCount,
                        0,
                        gradient.ColumnCount
                    );

                    matrix.At(j, 0, Config.LearningRate * neuronBackward.Row(0).At(0));

                    for (int k = 0; k < currentLayer.InputCount; k++)
                    {
                        matrix.At(j, k + 1, Config.LearningRate * neuronBackward.Row(k).At(1));
                    }
                }

                gradients.Add(matrix);
            }
        }

        protected class TrainState
        {
            public int index;
            public Vector<double> input;
            public Vector<double> target;
        }

        /// <summary>
        /// Run teaching epoch.
        /// </summary>
        /// <returns></returns>
        public override double RunEpoch()
        {
            var data = DataTransformer.Transform(Path(Config.Train.Data));

            int[] permutation = null;

            switch (Config.Batch)
            {
                case NetworkConfig.BatchType.Full:
                    {
                        permutation = Combinatorics.GeneratePermutation(data.RowCount);
                        break;
                    }
                case NetworkConfig.BatchType.Mini:
                    {
                        var batchSize = data.RowCount;

                        if (Config.BatchSize > 0 && Config.BatchSize <= data.RowCount)
                        {
                            batchSize = Config.BatchSize;
                        }

                        permutation = Combinatorics.GenerateVariation(data.RowCount, batchSize);
                        break;
                    }
                default:
                    {
                        permutation = new int[] { DiscreteUniform.Sample(0, data.RowCount - 1) };
                        break;
                    }
            }

            if (data.ColumnCount != Config.Inputs)
            {
                throw new Exception("Invalid data format.");
            }

            accumulatedCost = 0;

            accumulatedGradient = new Matrix<double>[layers.Count];

            var tasks = new Task[permutation.Length];

            for (int i = 0; i < permutation.Length; i++)
            {
                var index = permutation[i];

                tasks[i] = Task.Factory.StartNew((object trainState) =>
                {
                    var state = trainState as TrainState;

                    Train(
                        state.input,
                        state.target,
                        out double cost,
                        out List<Matrix<double>> gradient
                    );

                // Console.WriteLine("\nTask cost {0} + {1}", cost, accumulatedCost);

                accumulatedCost += cost;

                    for (int g = 0; g < gradient.Count; g++)
                    {
                        if (accumulatedGradient[g] == null)
                        {
                            accumulatedGradient[g] = Matrix<double>.Build.Dense(gradient[g].RowCount, gradient[g].ColumnCount);
                        }
                        accumulatedGradient[g] += gradient[g];
                    }
                }, new TrainState
                {
                    input = data.Row(index),
                    target = LabelTransformer.TransformLabels().Row(index)
                }, TaskCreationOptions.LongRunning);

                //tasks[i].Wait(1);
                /*
                Train(
                    data.Row(index),
                    LabelTransformer.TransformLabels().Row(index),
                    out double cost,
                    out List<Matrix<double>> gradient
                );

                accumulatedCost += cost;

                for (int g = 0; g < gradient.Count; g++)
                {
                    if (accumulatedGradient[g] == null)
                    {
                        accumulatedGradient[g] = Matrix<double>.Build.Dense(gradient[g].RowCount, gradient[g].ColumnCount);
                    }
                    accumulatedGradient[g] += gradient[g];
                }
                */
            }

            Task.WaitAll(tasks, Timeout.Infinite, CancellationToken.None);

            // Go through each layer of the network in opposite direction.
            for (int i = layers.Count - 1; i >= 0; i--)
            {
                var layer = layers[i];
                var layerGradient = accumulatedGradient[layers.Count - 1 - i];

                if (Program.Debug)
                {
                    Console.WriteLine("accumulated gradient for layer {0}: {1}", i, layerGradient);
                }

                for (int j = 0; j < layer.Size; j++)
                {
                    for (int k = 0; k < layer.Neurons[j].InputCount; k++)
                    {
                        var v = layerGradient.Row(j);
                        layer.Neurons[j].Weights[k] += layerGradient.Row(j).At(k + 1);
                    }
                    layer.Neurons[j].Bias += layerGradient.Row(j).At(0);
                }
            }

            return accumulatedCost / permutation.Length;
        }
    }
}
