using System;

namespace ML.Network.ActivationFunction
{
    class ReLU : IActivationFunction
    {
        /// <summary>
        /// Calculate the result of the function.
        /// </summary>
        /// <param name="net"></param>
        /// <returns></returns>
        public double Calculate(double net)
        {
            return Math.Max(0, net);
        }

        /// <summary>
        /// Calculate the derivative of the function.
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public double Derivative(double x)
        {
            // not differentiable at zero, make
            // it 1 instead to make gradient pass
            // (this case is very uncommon).
            if (x >= 0)
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
