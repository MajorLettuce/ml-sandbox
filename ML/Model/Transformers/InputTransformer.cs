namespace ML.Model.Transformers
{
    abstract class InputTransformer : DataTransformer
    {
        public InputTransformer(NetworkModel model) : base(model) { }
    }
}
