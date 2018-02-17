using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommandLine;
using ML.Model;
using ML.Network.ActivationFunction;
using MathNet.Numerics.LinearAlgebra;
using OxyPlot;
using OxyPlot.WindowsForms;

namespace ML
{
    class Program
    {
        static void Main(string[] args)
        {
            NetworkModel model = null;
            bool teaching = false;
            int epochRuns = 1;

            CommandLine.Parser.Default.ParseArguments<Options>(args)
                .WithParsed<Options>(opts =>
                {
                    teaching = opts.Teaching;
                    epochRuns = opts.EpochRuns;

#if !DEBUG
                    try
                    {
#endif
                    model = NetworkModel.Load(opts.Model);
#if !DEBUG

                        Console.Clear();
                    }
                    catch (Exception e)
                    {
                        Console.WriteLine(e.Message);
                        Environment.Exit(1);
                    }
#endif
                });

            if (teaching)
            {
                Console.WriteLine("Launching model in teaching mode");

                var plotModel = new PlotModel { Title = String.Format("{0} model \"{1}\" learning graph", model.Config.Type, model.Name) };
                var series = new OxyPlot.Series.LineSeries();
                plotModel.Series.Add(series);
                plotModel.Axes.Add(new OxyPlot.Axes.LinearAxis
                {
                    Minimum = 1,
                    Position = OxyPlot.Axes.AxisPosition.Bottom,
                    Title = "Epochs",
                });
                plotModel.Axes.Add(new OxyPlot.Axes.LinearAxis
                {
                    Position = OxyPlot.Axes.AxisPosition.Left,
                    Title = "Error",
                });

                var epoch = 0;

                var epochRunStep = epochRuns;

                while (epochRuns == 0 || epoch < epochRuns)
                {
                    Console.Write("Epoch {0}: ", ++epoch);

                    var previous = model.GetInfo();

                    var error = model.RunEpoch();
                    /*
                    Console.WriteLine("Previous model:");
                    Console.WriteLine("===========");
                    Console.WriteLine(previous);

                    Console.WriteLine("\nNew model:");
                    Console.WriteLine("===========");
                    Console.WriteLine(model.GetInfo());
                    */
                    Console.Write("error {0} ", error);

                    series.Points.Add(new DataPoint(epoch, error));

                    if (model.Config.Threshold != null && Math.Abs(error) <= model.Config.Threshold)
                    {
                        Console.WriteLine("\nError threshold ({0}) reached. Finished learning.", model.Config.Threshold);
                        Environment.Exit(1);
                    }

                    if (epochRuns == 0 || epoch >= epochRuns)
                    {
                        epochRuns += epochRunStep;
                        var pngExporter = new PngExporter();
                        pngExporter.ExportToFile(plotModel, model.Path("learning-graph.png"));

                        DisplayActions(model);
                    }
                }
            }
            else
            {
                Console.WriteLine("Running execution loop\n");

                while (true)
                {
                    Console.WriteLine("Model info:");
                    Console.WriteLine("===========");
                    Console.WriteLine(model.GetInfo());

                    DisplayActions(model);

                    var inputs = Vector<double>.Build.Dense(model.Config.Inputs);

                    for (int i = 0; i < model.Config.Inputs; i++)
                    {
                        inputs[i] = ReadDouble(String.Format("Input[{0}]", i));
                    }

                    Console.Clear();

                    Console.WriteLine("Inputs:");
                    Console.WriteLine("=======");
                    Console.WriteLine(inputs.ToVectorString());

                    Console.WriteLine("Result:");
                    Console.WriteLine("=======");
                    Console.WriteLine(model.Process(inputs).ToVectorString());
                }
            }
        }

        static void DisplayActions(NetworkModel model)
        {
            Console.WriteLine("\nActions:");
            Console.WriteLine("[Any key] Run | [Q]uit | [S]ave | [D]elete");
            Console.WriteLine("");

            switch (Console.ReadKey(true).Key)
            {
                case ConsoleKey.Q:
                    {
                        Environment.Exit(0);
                        break;
                    }
                case ConsoleKey.S:
                    {
                        Console.WriteLine("Saving model...");
                        model.Save();
                        Console.WriteLine("Model has been saved.");
                        Environment.Exit(0);
                        break;
                    }
                case ConsoleKey.D:
                    {
                        Console.WriteLine("Deleting model...");
                        model.Delete();
                        Console.WriteLine("Model has been deleted.");
                        Environment.Exit(0);
                        break;
                    }
            }
        }

        /// <summary>
        /// Read input and try convert it to double.
        /// Prompt user again when convertion fails.
        /// </summary>
        /// <param name="label"></param>
        /// <returns></returns>
        static double ReadDouble(string label)
        {
            while (true)
            {
                Console.Write("{0}: ", label);
                try
                {
                    return Convert.ToDouble(Console.ReadLine());
                }
                catch
                {
                    Console.WriteLine("Incorrect format.");
                }
            }
        }
    }
}
