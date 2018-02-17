namespace ML.Network.ActivationFunction
{
    class Linear : IActivationFunction
    {
        /// <summary>
        /// Calculate the result of the function.
        /// </summary>
        /// <param name="net"></param>
        /// <returns></returns>
        public double Calculate(double net)
        {
            return net;
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
