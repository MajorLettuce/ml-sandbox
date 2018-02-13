using Newtonsoft.Json;

namespace ML.Model
{
    class Config
    {
        [JsonProperty]
        public string Type { get; set; }

        [JsonProperty]
        public int Inputs { get; set; }
    }
}
