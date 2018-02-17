﻿using System;
using System.IO;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using Newtonsoft.Json;

namespace ML.Model
{
    abstract class NetworkModel
    {
        /// <summary>
        /// Model configuration instance.
        /// </summary>
        public Config Config { get; protected set; }

        /// <summary>
        /// Model name.
        /// </summary>
        public string Name { get; protected set; }

        /// <summary>
        /// Name of the model configuration file.
        /// </summary>
        static protected string file = "model.json";

        /// <summary>
        /// Default machine learning model constructor.
        /// </summary>
        /// <param name="model"></param>
        /// <param name="config"></param>
        public NetworkModel(string model, Config config)
        {
            Name = model;
            Config = config;
        }

        /// <summary>
        /// Load model using model configuration file.
        /// </summary>
        /// <param name="model"></param>
        /// <returns></returns>
        static public NetworkModel Load(string model)
        {
            var config = ReadConfig(model);

            var type = Type.GetType(String.Format("ML.Model.{0}", config.Type));
            if (type == null)
            {
                throw new Exception("Model type '" + config.Type + "' for model '" + model + "' doesn't exist.");
            }
            else
            {
                var configType = Type.GetType(
                    String.Format("ML.Model.{0}Config", config.Type.ToString().Split('.').Last())
                );

                object typeConfig = config;

                if (configType != null)
                {
                    try
                    {
                        typeConfig = ReadConfig(model, configType);
                    }
                    catch (Exception e)
                    {
                        throw e.InnerException;
                    }
                }

                try
                {
                    return Activator.CreateInstance(type, model, typeConfig) as NetworkModel;
                }
                catch (Exception e)
                {
                    throw new Exception("Unable to create model '" + model + "'.", e);
                }
            }
        }

        /// <summary>
        /// Initialize current model.
        /// </summary>
        abstract public void Initialize();

        /// <summary>
        /// Save currently loaded model.
        /// </summary>
        abstract public void Save();

        /// <summary>
        /// Delete currently loaded model.
        /// </summary>
        abstract public void Delete();

        /// <summary>
        /// File path relative to the model folder.
        /// </summary>
        /// <param name="model"></param>
        /// <param name="path"></param>
        /// <returns></returns>
        static public string Path(string model, string path)
        {
            return String.Format("models/{0}/{1}", model, path);
        }

        /// <summary>
        /// File path relative to the model folder.
        /// </summary>
        /// <param name="path"></param>
        /// <returns></returns>
        public string Path(string path)
        {
            return String.Format("models/{0}/{1}", Name, path);
        }

        /// <summary>
        /// Read plain-text resource from model folder.
        /// </summary>
        /// <param name="model"></param>
        /// <param name="path"></param>
        /// <returns></returns>
        static public string ReadResource(string model, string path)
        {
            return File.ReadAllText(Path(model, path));
        }

        /// <summary>
        /// Read plain-text resource from model folder.
        /// </summary>
        /// <param name="path"></param>
        /// <returns></returns>
        public string ReadResource(string path)
        {
            return File.ReadAllText(Path(Name, path));
        }

        /// <summary>
        /// Read generic model config without model specific properties.
        /// </summary>
        /// <param name="model"></param>
        /// <returns></returns>
        static public Config ReadConfig(string model)
        {
            return JsonConvert.DeserializeObject<Config>(ReadResource(model, file));
        }

        /// <summary>
        /// Read model type specific config.
        /// </summary>
        /// <param name="model"></param>
        /// <param name="type"></param>
        /// <returns></returns>
        static public object ReadConfig(string model, Type type)
        {
            return JsonConvert.DeserializeObject(ReadResource(model, file), type);
        }

        /// <summary>
        /// Get model info as text string.
        /// </summary>
        /// <returns></returns>
        virtual public string GetInfo()
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Process given inputs through the model.
        /// </summary>
        /// <param name="inputs"></param>
        /// <returns></returns>
        abstract public Vector<double> Process(Vector<double> inputs);

        /// <summary>
        /// Run teaching interation.
        /// </summary>
        /// <returns></returns>
        abstract public double Teach(Vector<double> inputs, Vector<double> outputs);

        /// <summary>
        /// Run teaching epoch.
        /// </summary>
        /// <returns></returns>
        abstract public double RunEpoch();
    }
}
