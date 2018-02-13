namespace ML.Network.ActivationFunction
{
    class Hardlim : IActivationFunction
    {
        public double Calculate(double net)
        {
            if (net <= 0)
            {
                return 0;
            }
            else if (net > 1)
            {
                return 1;
            }
            else
            {
                return net;
            }
        }
    }
}
