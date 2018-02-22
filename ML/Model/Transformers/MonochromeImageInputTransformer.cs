using System;
using System.Drawing;
using MathNet.Numerics.LinearAlgebra;

namespace ML.Model.Transformers
{
    class MonochromeImageInputTransformer : InputTransformer
    {
        public MonochromeImageInputTransformer(NetworkModel model) : base(model) { }

        /// <summary>
        /// Transform data file into array of vectors.
        /// </summary>
        /// <param name="file"></param>
        /// <returns></returns>
        public override Matrix<double> Transform(string file)
        {
            var bitmap = new Bitmap(Image.FromFile(file));

            var vector = Vector<double>.Build.Dense(bitmap.Width * bitmap.Height, index =>
            {
                return bitmap.GetPixel(index % bitmap.Width, index / bitmap.Height).GetBrightness();
            });

            return Matrix<double>.Build.DenseOfRowVectors(vector);
        }
    }
}
