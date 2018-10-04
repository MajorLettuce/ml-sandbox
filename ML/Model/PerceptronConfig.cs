using Newtonsoft.Json;
using System.ComponentModel;
using Newtonsoft.Json.Converters;

namespace ML.Model
{
    class PerceptronConfig : Config
    {
        public enum ActivationFunction
        {
            Heaviside,
            Hardlim,
            Linear,
        }

        [JsonProperty]
        [JsonConverter(typeof(StringEnumConverter))]
        public ActivationFunction Function { get; set; }

        [JsonProperty(DefaultValueHandling = DefaultValueHandling.Populate)]
        [DefaultValue("state.csv")]
        public string State { get; set; }

        [JsonProperty(DefaultValueHandling = DefaultValueHandling.Populate)]
        [DefaultValue("samples.csv")]
        public string Samples { get; set; }
    }
}
