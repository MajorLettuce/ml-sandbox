using Newtonsoft.Json;
using System.ComponentModel;

namespace ML.Model
{
    class TrainConfig
    {
        [JsonProperty(DefaultValueHandling = DefaultValueHandling.Populate)]
        [DefaultValue("train-data")]
        public string Data { get; set; }

        [JsonProperty(DefaultValueHandling = DefaultValueHandling.Populate)]
        [DefaultValue("train-labels")]
        public string Labels { get; set; }
    }
}
