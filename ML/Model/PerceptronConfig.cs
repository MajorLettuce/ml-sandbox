using Newtonsoft.Json;
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
    }
}
