using Newtonsoft.Json;
using Newtonsoft.Json.Converters;
using System.ComponentModel;

namespace ML.Model
{
    class Config
    {
        public enum Types
        {
            Perceptron,
            Network
        }

        [JsonProperty]
        [JsonConverter(typeof(StringEnumConverter))]
        public Types Type { get; set; }

        [JsonProperty]
        public int Inputs { get; set; }

        [JsonProperty(DefaultValueHandling = DefaultValueHandling.Populate)]
        [DefaultValue(0.01)]
        public double LearningRate { get; set; }

        [JsonProperty]
        public double? Threshold { get; set; }

        [JsonProperty]
        public TransformersConfig Transformers { get; set; }

        [JsonProperty(DefaultValueHandling = DefaultValueHandling.Populate)]
        [DefaultValue("train-data")]
        public string TrainingData { get; set; }

        [JsonProperty(DefaultValueHandling = DefaultValueHandling.Populate)]
        [DefaultValue("train-labels")]
        public string TrainLabels { get; set; }

        [JsonProperty(DefaultValueHandling = DefaultValueHandling.Populate)]
        [DefaultValue("test-data")]
        public string TestData { get; set; }

        [JsonProperty(DefaultValueHandling = DefaultValueHandling.Populate)]
        [DefaultValue("test-labels")]
        public string TestLabels { get; set; }
    }
}
