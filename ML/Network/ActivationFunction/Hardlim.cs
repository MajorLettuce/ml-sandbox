namespace ML.Network.ActivationFunction
{
    class Hardlim : IActivationFunction
    {
        /// <summary>
        /// Calculate the result of the function.
        /// </summary>
        /// <param name="net"></param>
        /// <returns></returns>
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

        /// <summary>
        /// Calculate the derivative of the function.
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public double Derivative(double x)
        {
            return 0;
        }
    }
}
