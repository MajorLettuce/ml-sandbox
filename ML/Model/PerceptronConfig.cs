using Newtonsoft.Json;
using Newtonsoft.Json.Converters;

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
    }
}
