using CommandLine;

namespace ML
{
    class Options
    {
        [Option('t', "teach", Default = false, HelpText = "Run model in teaching mode.")]
        public bool Teaching { get; set; }

        [Option('m', "model", Default = "default", HelpText = "Name of the model to load.")]
        public string Model { get; set; }

        [Option("runs", Default = 1, HelpText = "Number of runs to run for each epoch.")]
        public int EpochRuns { get; set; }
    }
}
