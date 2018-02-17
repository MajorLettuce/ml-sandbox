namespace ML.Network.ActivationFunction
{
    interface IActivationFunction
    {
        /// <summary>
        /// Calculate the result of the function.
        /// </summary>
        /// <param name="net"></param>
        /// <returns></returns>
        double Calculate(double net);

        /// <summary>
        /// Calculate the derivative of the function.
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        double Derivative(double x);
    }
}
