using Newtonsoft.Json;
using Newtonsoft.Json.Converters;
using System.ComponentModel;
using System.Collections.Generic;

namespace ML.Model
{
    class NetworkConfig : Config
    {
        [JsonProperty]
        public List<LayerConfig> Layers { get; set; }

        public enum BatchType
        {
            None,
            Mini,
            Full,
        }

        [JsonProperty]
        [JsonConverter(typeof(StringEnumConverter))]
        public BatchType Batch { get; set; }

        [JsonProperty]
        [DefaultValue(32)]
        public int BatchSize { get; set; }
    }
}
