using System;

namespace ML.Network.ActivationFunction
{
    class Sigmoid : IActivationFunction
    {
        /// <summary>
        /// Calculate the result of the function.
        /// </summary>
        /// <param name="net"></param>
        /// <returns></returns>
        public double Calculate(double net)
        {
            return 1 / (1 + Math.Exp(-net));
        }

        /// <summary>
        /// Calculate the derivative of the function.
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public double Derivative(double x)
        {
            return Calculate(x) * (1 - Calculate(x));
        }
    }
}
