import {Array2D, NDArrayMathGPU, Scalar} from 'deeplearn';

export default () => {
  const math = new NDArrayMathGPU();

  const a = Array2D.new([2, 2], [1.0, 2.0, 3.0, 4.0]);
  const b = Array2D.new([2, 2], [0.0, 2.0, 4.0, 6.0]);

  // Non-blocking math calls.
  const diff = math.sub(a, b);
  const squaredDiff = math.elementWiseMul(diff, diff);
  const sum = math.sum(squaredDiff);
  const size = Scalar.new(a.size);
  const average = math.divide(sum, size);

  console.log('mean squared difference: ' + average.get());
}

