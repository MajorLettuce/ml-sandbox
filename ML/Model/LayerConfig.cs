using Newtonsoft.Json;
using Newtonsoft.Json.Converters;

namespace ML.Model
{
    class LayerConfig
    {
        public enum LayerType
        {
            FC, // Fully-connected layer
        }

        [JsonProperty]
        [JsonConverter(typeof(StringEnumConverter))]
        public LayerType Type { get; set; }

        [JsonProperty(PropertyName = "Neurons")]
        public int NeuronCount { get; set; }

        public enum ActivationFunction
        {
            Sigmoid,
            ReLU,
            Softmax,
        }

        [JsonProperty]
        [JsonConverter(typeof(StringEnumConverter))]
        public ActivationFunction Function { get; set; }
    }
}
