using Newtonsoft.Json;
using Newtonsoft.Json.Converters;

namespace ML.Model
{
    class TransformersConfig
    {
        public enum InputTransformerType
        {
            Vector,
        }

        [JsonProperty]
        [JsonConverter(typeof(StringEnumConverter))]
        public InputTransformerType Input { get; set; }

        public enum OutputTransformerType
        {
            Vector,
        }

        [JsonProperty]
        [JsonConverter(typeof(StringEnumConverter))]
        public OutputTransformerType Output { get; set; }
    }
}
