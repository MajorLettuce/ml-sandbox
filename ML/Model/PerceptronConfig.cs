using Newtonsoft.Json;
using Newtonsoft.Json.Converters;
using System.ComponentModel;

namespace ML.Model
{
    class PerceptronConfig : Config
    {
        public enum ActivationFunctions
        {
            Heaviside,
            Hardlim
        }

        [JsonProperty]
        [JsonConverter(typeof(StringEnumConverter))]
        public ActivationFunctions Function { get; set; }

        [DefaultValue(0.01)]
        [JsonProperty(DefaultValueHandling = DefaultValueHandling.Populate)]
        public double LearningRate { get; set; }
    }
}
