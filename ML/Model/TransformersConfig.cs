using Newtonsoft.Json;
using Newtonsoft.Json.Converters;

namespace ML.Model
{
    class TransformersConfig
    {
        public enum DataTransformerType
        {
            Vector,
        }

        [JsonProperty]
        [JsonConverter(typeof(StringEnumConverter))]
        public DataTransformerType Data { get; set; }

        public enum LabelTransformerType
        {
            Vector,
        }

        [JsonProperty]
        [JsonConverter(typeof(StringEnumConverter))]
        public LabelTransformerType Label { get; set; }
    }
}
