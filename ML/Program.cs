﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommandLine;
using ML.Model;
using ML.Network.ActivationFunction;
using MathNet.Numerics.LinearAlgebra;

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

                uint epoch = 1;

                while (true)
                {
                    Console.Clear();
                    Console.WriteLine("Running learning epoch number {0}...\n", epoch++);

                    var previous = model.GetInfo();

                    for (int run = 0; run < epochRuns; run++)
                    {
                        model.RunEpoch();
                    }

                    Console.WriteLine("Done.\n");

                    Console.WriteLine("Previous model:");
                    Console.WriteLine("===========");
                    Console.WriteLine(previous);

                    Console.WriteLine("\nNew model:");
                    Console.WriteLine("===========");
                    Console.WriteLine(model.GetInfo());

                    DisplayActions(model);
                }
            }
            else
            {
                Console.WriteLine("Running execution loop\n");

                bool debug = true;

                while (true)
                {
                    if (debug)
                    {
                        Console.WriteLine("Model info:");
                        Console.WriteLine("===========");
                        Console.WriteLine(model.GetInfo());
                    }

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
                        model.Save();
                        Console.WriteLine("Model has been saved.");
                        Environment.Exit(0);
                        break;
                    }
                case ConsoleKey.D:
                    {
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