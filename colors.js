/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

// tslint:disable-next-line:max-line-length
import {Array1D, CostReduction, FeedEntry, Graph, InCPUMemoryShuffledInputProviderBuilder, NDArrayMath, NDArrayMathGPU, Session, SGDOptimizer, Tensor} from 'deeplearn';

class ComplementaryColorModel {
  // Runs training.
  session;

  // Encapsulates math operations on the CPU and GPU.
  math = new NDArrayMathGPU();

  // An optimizer with a certain initial learning rate. Used for training.
  initialLearningRate = 0.042;
  optimizer;

  // Each training batch will be on this many examples.
  batchSize = 300;

  inputTensor;
  targetTensor;
  costTensor;
  predictionTensor;

  // Maps tensors to InputProviders.
  feedEntries;

  constructor() {
    this.optimizer = new SGDOptimizer(this.initialLearningRate);
  }

  /**
   * Constructs the graph of the model. Call this method before training.
   */
  setupSession() {
    const graph = new Graph();

    // This tensor contains the input. In this case, it is a scalar.
    this.inputTensor = graph.placeholder('input', [5]);

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
        this.createFullyConnectedLayer(graph, fullyConnectedLayer, 3, 1);

    // We will optimize using mean squared loss.
    this.costTensor =
        graph.meanSquaredCost(this.targetTensor, this.predictionTensor);

    // Create the session only after constructing the graph.
    this.session = new Session(graph, this.math);

    // Generate the data that will be used to train the model.
    this.generateTrainingData();
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
  train1Batch(shouldFetchCost) {
    // Every 42 steps, lower the learning rate by 15%.
    const learningRate =
        this.initialLearningRate * Math.pow(0.85, Math.floor(step / 42));
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

  predict(arr) {
    let values;
    this.math.scope((keep, track) => {
      const mapping = [{
        tensor: this.inputTensor,
        data: Array1D.new(arr),
      }];
      const evalOutput = this.session.eval(this.predictionTensor, mapping);
      values = evalOutput.getValues();
    });
    return values;
  }

  createFullyConnectedLayer(
      graph, inputLayer, layerIndex,
      sizeOfThisLayer) {
    return graph.layers.dense(
        'fully_connected_' + layerIndex, inputLayer, sizeOfThisLayer,
        (x) => graph.relu(x));
  }

  /**
   * Generates data used to train. Creates a feed entry that will later be used
   * to pass data into the model. Generates `exampleCount` data points.
   */
  generateTrainingData() {
    this.math.scope(() => {

      // Store the data within Array1Ds so that learnjs can use it.
      const inputArray = [
        Array1D.new([0,0,0,0,0]),
        Array1D.new([0,1,0,0,0]),
        Array1D.new([0,0,2,0,0]),
        Array1D.new([0,0,0,3,0]),
        Array1D.new([0,0,0,0,4])
      ];
      const targetArray = [
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
              [inputArray, targetArray]);
      const [inputProvider, targetProvider] =
          shuffledInputProviderBuilder.getInputProviders();

      // Maps tensors to InputProviders.
      this.feedEntries = [
        {tensor: this.inputTensor, data: inputProvider},
        {tensor: this.targetTensor, data: targetProvider}
      ];
    });
  }
}

const complementaryColorModel = new ComplementaryColorModel();
window.complementaryColorModel = complementaryColorModel;

// Create the graph of the model.
complementaryColorModel.setupSession();

// On every frame, we train and then maybe update the UI.
let step = 0;
function trainAndMaybeRender() {
  if (step > 50) {
    // Stop training.
    return;
  }

  // Schedule the next batch to be trained.
  requestAnimationFrame(trainAndMaybeRender);

  // We only fetch the cost every 5 steps because doing so requires a transfer
  // of data from the GPU.
  const localStepsToRun = 5;
  let cost;
  for (let i = 0; i < localStepsToRun; i++) {
    cost = complementaryColorModel.train1Batch(true);
    step++;
  }

  // Print data to console so the user can inspect.
  console.log('step', step - 1, 'cost', cost);

}

trainAndMaybeRender();
