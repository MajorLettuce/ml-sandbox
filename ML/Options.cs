using CommandLine;

namespace ML
{
    class Options
    {
        [Option('d', "debug", Default = false, HelpText = "Enable debug mode.")]
        public bool Debug { get; set; }

        [Option('t', "teach", Default = false, HelpText = "Run model in teaching mode.")]
        public bool Teaching { get; set; }

        [Option('m', "model", Default = "default", HelpText = "Name of the model to load.")]
        public string Model { get; set; }

        [Option("runs", Default = 0, HelpText = "Number of epochs to run. (0 - unlimited)")]
        public int EpochRuns { get; set; }
    }
}
