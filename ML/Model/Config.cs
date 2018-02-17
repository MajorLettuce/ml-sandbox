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
    }
}
