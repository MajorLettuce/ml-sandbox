using System.Collections.Generic;
using Newtonsoft.Json;

namespace ML.Model
{
    class NetworkConfig : Config
    {
        [JsonProperty]
        public List<LayerConfig> Layers { get; set; }
    }
}
