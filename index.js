import {Array1D, NDArrayMathGPU, Scalar} from 'deeplearn';

const math = new NDArrayMathGPU();
const a = Array1D.new([1, 2, 3]);
const b = Scalar.new(2);

const result = math.add(a, b);
// Float32Array([3, 4, 5])
result.getValuesAsync().then(values => console.log(values));
