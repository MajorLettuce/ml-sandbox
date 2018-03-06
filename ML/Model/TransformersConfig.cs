using Newtonsoft.Json;
using Newtonsoft.Json.Converters;

namespace ML.Model
{
    class TransformersConfig
    {
        public enum InputTransformerType
        {
            MonochromeImage,
        }

        [JsonProperty]
        [JsonConverter(typeof(StringEnumConverter))]
        public InputTransformerType Input { get; set; }

        public enum DataTransformerType
        {
            Vector,
            Mnist,
            MonochromeImage,
        }

        [JsonProperty]
        [JsonConverter(typeof(StringEnumConverter))]
        public DataTransformerType Data { get; set; }

        public enum LabelTransformerType
        {
            Vector,
            Single,
            Mnist,
            Top,
        }

        [JsonProperty]
        [JsonConverter(typeof(StringEnumConverter))]
        public LabelTransformerType Label { get; set; }
    }
}
