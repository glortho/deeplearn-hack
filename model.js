import ndarray from 'ndarray';
import SphericalMercator from '@mapbox/sphericalmercator';
import unpack from 'ndarray-unpack';
import imshow from 'ndarray-imshow';

const merc = new SphericalMercator({
  size: 256
});

import {GPGPUContext, gpgpu_util, Array3D, Array1D, CostReduction, FeedEntry, Graph, InCPUMemoryShuffledInputProviderBuilder, NDArrayMath, NDArrayMathGPU, Session, SGDOptimizer, Tensor, Scalar} from 'deeplearn';

import {
  addTraining as addTrainingToDb,
  clear as clearDb
} from './db';


class Model {
  // Runs training.
  session;

  // Encapsulates math operations on the CPU and GPU.
  math = new NDArrayMathGPU();

  // An optimizer with a certain initial learning rate. Used for training.
  initialLearningRate = 0.042;
  optimizer;

  batchSize;

  inputTensor;
  targetTensor;
  costTensor;
  predictionTensor;

  // Maps tensors to InputProviders.
  feedEntries;

  inputSize = 3000

  constructor() {
    this.optimizer = new SGDOptimizer(this.initialLearningRate);
  }

  /**
   * Constructs the graph of the model. Call this method before training.
   */
  setupSession() {
    const graph = new Graph();

    // This tensor contains the input. In this case, it is a scalar.
    this.inputTensor = graph.placeholder('input', [this.inputSize]);

    // This tensor contains the target.
    this.targetTensor = graph.placeholder('output', [1]);

    // Create 3 fully connected layers, each with half the number of nodes of
    // the previous layer. The first one has 64 nodes.
    let fullyConnectedLayer =
        this.createFullyConnectedLayer(graph, this.inputTensor, 0, 64);

    // Create fully connected layer 1, which has 32 nodes.
    fullyConnectedLayer =
        this.createFullyConnectedLayer(graph, fullyConnectedLayer, 1, 32);

    // Create fully connected layer 2, which has 16 nodes.
    fullyConnectedLayer =
        this.createFullyConnectedLayer(graph, fullyConnectedLayer, 2, 16);

    this.predictionTensor =
        this.createFullyConnectedLayer(graph, fullyConnectedLayer, 4, 1);

    // We will optimize using mean squared loss.
    this.costTensor =
        graph.meanSquaredCost(this.targetTensor, this.predictionTensor);

    this.graph = graph;

    // Create the session only after constructing the graph.
    this.session = new Session(graph, this.math);

    // Generate the data that will be used to train the model.
    this.inputArray = [];
    this.targetArray = [];
    //this.generateTrainingData();
  }

  unpack_flat = (view) => {
    const arr = unpack(view)
      .reduce((master, row) => master.concat( row ), [] )
      .filter( item => !!item || item === 0 );

    //return arr;
    const size = this.inputSize;
    const arrNew = arr.length < size
      ? arr.concat( Array( size - arr.length ).fill( 0 ) )
      : arr.slice(0, size);
    return arrNew;
  }

  tilePx = (lat, lon, tile_bbox) => {
    const w = 256;
    const h = 256;
    const lat0 = tile_bbox[3];
    const lon0 = tile_bbox[0];
    const latD = tile_bbox[3] - tile_bbox[1];
    const lonD = tile_bbox[2] - tile_bbox[0];
    let longitude = lon;
    longitude -= lon0;
    let latitude = lat0 - lat;
    const x = (w*(longitude/lonD));
    const y = Math.min(256.0, (h*(latitude/latD)));
    return [parseInt(Math.max(0,x)), parseInt(Math.max(0,y))];
  }

  /**
   * Trains one batch for one iteration. Call this method multiple times to
   * progressively train. Calling this function transfers data from the GPU in
   * order to obtain the current loss on training data.
   *
   * If shouldFetchCost is true, returns the mean cost across examples in the
   * batch. Otherwise, returns -1. We should only retrieve the cost now and then
   * because doing so requires transferring data from the GPU.
   */
  train1Batch(shouldFetchCost = false) {
    // Every 42 steps, lower the learning rate by 15%.
    const learningRate =
        this.initialLearningRate * Math.pow(0.85, Math.floor(this.step / 42));
    this.optimizer.setLearningRate(learningRate);

    // Train 1 batch.
    let costValue = -1;
    this.math.scope(() => {
      const cost = this.session.train(
          this.costTensor, this.feedEntries, this.batchSize, this.optimizer,
          shouldFetchCost ? CostReduction.MEAN : CostReduction.NONE);

      if (!shouldFetchCost) {
        // We only train. We do not compute the cost.
        return;
      }

      // Compute the cost (by calling get), which requires transferring data
      // from the GPU.
      costValue = cost.get();
    });
    return costValue;
  }


  predict( img, x, y ) {
    let values = false;
    this.math.scope((keep, track) => {
      const canvas = document.createElement('canvas')
      canvas.width = img.width
      canvas.height = img.height
      const context = canvas.getContext('2d')
      context.drawImage(img, 0, 0)
      const pixels = context.getImageData(0, 0, img.width, img.height);

      const arr = ndarray(new Uint8Array(pixels.data), [img.width, img.height, 4], [4*img.width, 4, 1], 0);
      //const clip = arr
      //  .hi(227, 227)

      for ( let xs=0; xs < 4; xs++ ) {
        for ( let ys=0; ys < 4; ys++ ) {
          const minX = xs * 64;
          const maxX = minX + 64;
          const minY = ys * 64;
          const maxY = minY + 64;
          const clip = arr
            .hi(maxY, maxX)
            .lo(minY, minX)

          //console.log(clip.shape)
          imshow(clip)

          const _red = this.unpack_flat(clip.pick(null, null, 0));
          const _green = this.unpack_flat(clip.pick(null, null, 1));
          const _blue = this.unpack_flat(clip.pick(null, null, 2));
          const rgb = [..._red, ..._green, ..._blue];

          const normalize = item => item/255

          const mapping = [{
            tensor: this.inputTensor,
            data: Array1D.new(_red.map( normalize )),
          }, {
            tensor: this.inputTensor,
            data: Array1D.new(_green.map( normalize )),
          }, {
            tensor: this.inputTensor,
            data: Array1D.new(_blue.map( normalize )),
          }];
          const evalOutput = this.session.eval(this.predictionTensor, mapping);
          const evals = evalOutput.getValues();
          console.log('PREDICT VAL', evals[0]);
        }
      }
    });
    return values;
  }

  createFullyConnectedLayer(
      graph, inputLayer, layerIndex,
      sizeOfThisLayer) {
    return graph.layers.dense(
        'fully_connected_' + layerIndex, inputLayer, sizeOfThisLayer,
        (x) => graph.relu(x), false);
  }

  /**
   * Generates data used to train. Creates a feed entry that will later be used
   * to pass data into the model. Generates `exampleCount` data points.
   */
  addTraining({ bbox, x, y, img, label, options = { addToDb: true } }) {
    this.math.scope(() => {
      const tile_bbox = merc.bbox(x, y, 18);
      const minXY = this.tilePx(bbox[3], bbox[0], tile_bbox);
      const maxXY = this.tilePx(bbox[1], bbox[2], tile_bbox);

      if ( (maxXY[1] - minXY[1]) > 0 && ( maxXY[0] - minXY[0] ) > 0 ) {

        const miny = minXY[1];
        const minx = minXY[0];
        const maxy = maxXY[1];
        const maxx = maxXY[0];

        const canvas = document.createElement('canvas')
        //img.width = img.width >= 227 ? img.width - ( img.width - 227 ) : img.width + ( 227 - img.width );
        //img.height = img.height >= 227 ? img.height - ( img.height - 227 ) : img.height + ( 227 - img.height );
        canvas.width = img.width
        canvas.height = img.height
        const context = canvas.getContext('2d')
        context.drawImage(img, 0, 0)
        const pixels = context.getImageData(0, 0, img.width, img.height);

        const arr = ndarray(new Uint8Array(pixels.data), [img.width, img.height, 4], [4*img.width, 4, 1], 0);

        const clip = arr
          .hi(maxy, maxx)
          .lo(miny, minx)

        const red = this.unpack_flat(clip.pick(null, null, 0));
        const green = this.unpack_flat(clip.pick(null, null, 1));
        const blue = this.unpack_flat(clip.pick(null, null, 2));
        //const rgb = [...red, ...green, ...blue];


        imshow(clip)

        if ( options.addToDb ) addTrainingToDb( { bbox, x, y, label } );

        if ( [ ...red, ...green, ...blue ].some( item => item !== 0 && !item ) )  {
          debugger;
          console.log(red, green, blue);
        }

        const normalize = item => item/255

        this.inputArray.push(
          Array1D.new( red.map( normalize ) ),
          Array1D.new( green.map( normalize ) ),
          Array1D.new( blue.map( normalize ) )
        );
        this.targetArray.push(
          Array1D.new( [ label ] ),
          Array1D.new( [ label ] ),
          Array1D.new( [ label ] )
        );

        const shuffledInputProviderBuilder =
          new InCPUMemoryShuffledInputProviderBuilder(
            [this.inputArray, this.targetArray]);
        const [inputProvider, targetProvider] =
          shuffledInputProviderBuilder.getInputProviders();

        // Maps tensors to InputProviders.
        this.feedEntries = [
          {tensor: this.inputTensor, data: inputProvider},
          {tensor: this.targetTensor, data: targetProvider}
        ];
      }
    });
  }

  generateTrainingData() {
    this.math.scope(() => {

      // Store the data within Array1Ds so that learnjs can use it.
      this.inputArray = [
        Array1D.new([0,0,0,0,0,0,0,0]),
        Array1D.new([0,1,0,0,0,0,0,1]),
        Array1D.new([0,0,2,0,0,0,0,2]),
        Array1D.new([0,0,0,3,0,0,0,3]),
        Array1D.new([0,0,0,0,4,0,0,4])
      ];
      this.targetArray = [
        Array1D.new([0]),
        Array1D.new([1]),
        Array1D.new([2]),
        Array1D.new([3]),
        Array1D.new([4])
      ];

      // This provider will shuffle the training data (and will do so in a way
      // that does not separate the input-target relationship).
      const shuffledInputProviderBuilder =
          new InCPUMemoryShuffledInputProviderBuilder(
              [this.inputArray, this.targetArray]);
      const [inputProvider, targetProvider] =
          shuffledInputProviderBuilder.getInputProviders();

      // Maps tensors to InputProviders.
      this.feedEntries = [
        {tensor: this.inputTensor, data: inputProvider},
        {tensor: this.targetTensor, data: targetProvider}
      ];
    });
  }

  trainLoop = () => {
    if (this.step > 100) {
      // Stop training.
      return;
    }

    // Schedule the next batch to be trained.
    requestAnimationFrame(this.trainLoop);

    // We only fetch the cost every 5 steps because doing so requires a transfer
    // of data from the GPU.
    const localStepsToRun = 5;
    let cost;
    for (let i = 0; i < localStepsToRun; i++) {
      cost = model.train1Batch(i === localStepsToRun - 1);
      this.step++;
    }

    // Print data to console so the user can inspect.
    console.log('step', this.step - 1, 'cost', cost);
  }

  train() {
    this.step = 0;
    this.batchSize = this.inputArray.length;
    this.trainLoop();
  }

  clear() {
    this.inputArray = [];
    this.targetArray = [];
    clearDb();
  }
}

const model = new Model();
model.setupSession();
window.model = model;
export default model;
