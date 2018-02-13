namespace ML.Network.ActivationFunction
{
    class Heaviside : IActivationFunction
    {
        public double Calculate(double net)
        {
            if (net >= 0)
            {
                return 1;
            }
            else
            {
                return 0;
            }
        }
    }
}
