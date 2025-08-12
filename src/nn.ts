/* Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

/**
 * A node in a neural network. Each node has a state
 * (total input, output, and their respectively derivatives) which changes
 * after every forward and back propagation run.
 */
export class Node {
  id: string;
  /** List of input links. */
  inputLinks: Link[] = [];
  bias = 0.1;
  /** List of output links. */
  outputs: Link[] = [];
  totalInput: number;
  output: number;
  /** Error derivative with respect to this node's output. */
  outputDer = 0;
  /** Error derivative with respect to this node's total input. */
  inputDer = 0;
  /**
   * Accumulated error derivative with respect to this node's total input since
   * the last update. This derivative equals dE/db where b is the node's
   * bias term.
   */
  accInputDer = 0;
  /**
   * Number of accumulated err. derivatives with respect to the total input
   * since the last update.
   */
  numAccumulatedDers = 0;
  /** Activation function that takes total input and returns node's output */
  activation: ActivationFunction;

  /**
   * Creates a new node with the provided id and activation function.
   */
  constructor(id: string, activation: ActivationFunction, initZero?: boolean) {
    this.id = id;
    this.activation = activation;
    if (initZero) {
      this.bias = 0;
    }
  }

  /** Recomputes the node's output and returns it. */
  updateOutput(): number {
    // Stores total input into the node.
    this.totalInput = this.bias;
    for (let j = 0; j < this.inputLinks.length; j++) {
      let link = this.inputLinks[j];
      this.totalInput += link.weight * link.source.output;
    }
    this.output = this.activation.output(this.totalInput);
    return this.output;
  }
}

/**
 * An error function and its derivative.
 */
export interface ErrorFunction {
  error: (output: number, target: number) => number;
  der: (output: number, target: number) => number;
}

/** A node's activation function and its derivative. */
export interface ActivationFunction {
  output: (input: number) => number;
  der: (input: number) => number;
}

/** Function that computes a penalty cost for a given weight in the network. */
export interface RegularizationFunction {
  output: (weight: number) => number;
  der: (weight: number) => number;
}

/** Built-in error functions */
export class Errors {
  public static SQUARE: ErrorFunction = {
    error: (output: number, target: number) =>
               0.5 * Math.pow(output - target, 2),
    der: (output: number, target: number) => output - target
  };
}

/** Polyfill for TANH */
(Math as any).tanh = (Math as any).tanh || function(x) {
  if (x === Infinity) {
    return 1;
  } else if (x === -Infinity) {
    return -1;
  } else {
    let e2x = Math.exp(2 * x);
    return (e2x - 1) / (e2x + 1);
  }
};

/** Built-in activation functions */
export class Activations {
  public static TANH: ActivationFunction = {
    output: x => (Math as any).tanh(x),
    der: x => {
      let output = Activations.TANH.output(x);
      return 1 - output * output;
    }
  };
  public static RELU: ActivationFunction = {
    output: x => Math.max(0, x),
    der: x => x <= 0 ? 0 : 1
  };
  public static SIGMOID: ActivationFunction = {
    output: x => 1 / (1 + Math.exp(-x)),
    der: x => {
      let output = Activations.SIGMOID.output(x);
      return output * (1 - output);
    }
  };
  public static LINEAR: ActivationFunction = {
    output: x => x,
    der: x => 1
  };
}

/** Build-in regularization functions */
export class RegularizationFunction {
  public static L1: RegularizationFunction = {
    output: w => Math.abs(w),
    der: w => w < 0 ? -1 : (w > 0 ? 1 : 0)
  };
  public static L2: RegularizationFunction = {
    output: w => 0.5 * w * w,
    der: w => w
  };
}

/**
 * A link in a neural network. Each link has a weight and a source and
 * destination node. Also it has an internal state (error derivative
 * with respect to a particular input) which gets updated after
 * a run of back propagation.
 */
export class Link {
  id: string;
  source: Node;
  dest: Node;
  weight = Math.random() - 0.5;
  isDead = false;
  /** Error derivative with respect to this weight. */
  errorDer = 0;
  /** Accumulated error derivative since the last update. */
  accErrorDer = 0;
  /** Number of accumulated derivatives since the last update. */
  numAccumulatedDers = 0;
  regularization: RegularizationFunction;

  /**
   * Constructs a link in the neural network initialized with random weight.
   *
   * @param source The source node.
   * @param dest The destination node.
   * @param regularization The regularization function that computes the
   *     penalty for this weight. If null, there will be no regularization.
   */
  constructor(source: Node, dest: Node,
      regularization: RegularizationFunction, initZero?: boolean) {
    this.id = source.id + "-" + dest.id;
    this.source = source;
    this.dest = dest;
    this.regularization = regularization;
    if (initZero) {
      this.weight = 0;
    }
  }
}

/** Type of layer supported by the network. */
export enum LayerType {
  DENSE = "dense",
  CONV = "conv",
  POOL = "pool"
}

/** Configuration for a convolutional layer. */
export interface ConvLayerConfig {
  type: LayerType.CONV;
  kernelSize: number;
  stride: number;
  padding: number;
  inDepth: number;
  outDepth: number;
  activation: ActivationFunction;
}

/** Configuration for a pooling layer. */
export interface PoolLayerConfig {
  type: LayerType.POOL;
  size: number;
  stride: number;
  padding: number;
  op: "max" | "avg";
}

/**
 * A simple convolutional layer with weight sharing. The implementation here is
 * intentionally minimal – it only supports 2D convolutions with square kernels
 * and unit batch size which is sufficient for the playground environment.
 */
export class ConvolutionLayer {
  type = LayerType.CONV;
  kernelSize: number;
  stride: number;
  padding: number;
  inDepth: number;
  outDepth: number;
  activation: ActivationFunction;
  regularization: RegularizationFunction;

  // Kernels indexed by [outDepth][inDepth][y][x]
  kernels: number[][][][] = [];
  biases: number[] = [];

  // Gradients for each kernel weight and bias.
  kernelGrads: number[][][][] = [];
  biasGrads: number[] = [];
  numAccumulatedDers = 0;

  // Last input and output for backprop.
  input: number[][][] = null;
  output: number[][][] = null;
  preActivation: number[][][] = null;

  constructor(config: ConvLayerConfig, regularization: RegularizationFunction) {
    this.kernelSize = config.kernelSize;
    this.stride = config.stride || 1;
    this.padding = config.padding || 0;
    this.inDepth = config.inDepth;
    this.outDepth = config.outDepth;
    this.activation = config.activation;
    this.regularization = regularization;

    for (let d = 0; d < this.outDepth; d++) {
      this.kernels[d] = [];
      this.kernelGrads[d] = [];
      for (let c = 0; c < this.inDepth; c++) {
        this.kernels[d][c] = [];
        this.kernelGrads[d][c] = [];
        for (let i = 0; i < this.kernelSize; i++) {
          this.kernels[d][c][i] = [];
          this.kernelGrads[d][c][i] = [];
          for (let j = 0; j < this.kernelSize; j++) {
            this.kernels[d][c][i][j] = Math.random() - 0.5;
            this.kernelGrads[d][c][i][j] = 0;
          }
        }
      }
      this.biases[d] = 0.1;
      this.biasGrads[d] = 0;
    }
  }

  forward(input: number[][][]): number[][][] {
    this.input = input;
    const inH = input.length;
    const inW = input[0].length;
    const inD = input[0][0].length;
    const outH = Math.floor((inH - this.kernelSize + 2 * this.padding) /
                                this.stride) + 1;
    const outW = Math.floor((inW - this.kernelSize + 2 * this.padding) /
                                this.stride) + 1;
    this.output = zeros3D(outH, outW, this.outDepth);
    this.preActivation = zeros3D(outH, outW, this.outDepth);
    for (let y = 0; y < outH; y++) {
      for (let x = 0; x < outW; x++) {
        for (let od = 0; od < this.outDepth; od++) {
          let sum = this.biases[od];
          for (let id = 0; id < inD; id++) {
            for (let ky = 0; ky < this.kernelSize; ky++) {
              for (let kx = 0; kx < this.kernelSize; kx++) {
                const inY = y * this.stride + ky - this.padding;
                const inX = x * this.stride + kx - this.padding;
                if (inY < 0 || inY >= inH || inX < 0 || inX >= inW) {
                  continue;
                }
                sum += this.kernels[od][id][ky][kx] *
                       input[inY][inX][id];
              }
            }
          }
          this.preActivation[y][x][od] = sum;
          this.output[y][x][od] = this.activation.output(sum);
        }
      }
    }
    return this.output;
  }

  backward(grad: number[][][]): number[][][] {
    const inH = this.input.length;
    const inW = this.input[0].length;
    const inD = this.input[0][0].length;
    const outH = this.output.length;
    const outW = this.output[0].length;
    const gradInput = zeros3D(inH, inW, inD);
    for (let y = 0; y < outH; y++) {
      for (let x = 0; x < outW; x++) {
        for (let od = 0; od < this.outDepth; od++) {
          const chain = grad[y][x][od] *
              this.activation.der(this.preActivation[y][x][od]);
          this.biasGrads[od] += chain;
          for (let id = 0; id < inD; id++) {
            for (let ky = 0; ky < this.kernelSize; ky++) {
              for (let kx = 0; kx < this.kernelSize; kx++) {
                const inY = y * this.stride + ky - this.padding;
                const inX = x * this.stride + kx - this.padding;
                if (inY < 0 || inY >= inH || inX < 0 || inX >= inW) {
                  continue;
                }
                this.kernelGrads[od][id][ky][kx] +=
                    chain * this.input[inY][inX][id];
                gradInput[inY][inX][id] +=
                    chain * this.kernels[od][id][ky][kx];
              }
            }
          }
        }
      }
    }
    this.numAccumulatedDers++;
    return gradInput;
  }

  getOutputShape(): [number, number, number] {
    return [this.output.length, this.output[0].length, this.outDepth];
  }

  updateWeights(learningRate: number, regularizationRate: number) {
    if (this.numAccumulatedDers === 0) {
      return;
    }
    for (let od = 0; od < this.outDepth; od++) {
      // Update biases.
      this.biases[od] -= learningRate *
          this.biasGrads[od] / this.numAccumulatedDers;
      this.biasGrads[od] = 0;
      for (let id = 0; id < this.inDepth; id++) {
        for (let ky = 0; ky < this.kernelSize; ky++) {
          for (let kx = 0; kx < this.kernelSize; kx++) {
            let grad = this.kernelGrads[od][id][ky][kx] /
                this.numAccumulatedDers;
            let w = this.kernels[od][id][ky][kx];
            w -= learningRate * grad;
            if (this.regularization) {
              w -= learningRate * regularizationRate *
                  this.regularization.der(this.kernels[od][id][ky][kx]);
            }
            this.kernels[od][id][ky][kx] = w;
            this.kernelGrads[od][id][ky][kx] = 0;
          }
        }
      }
    }
    this.numAccumulatedDers = 0;
  }
}

/**
 * A pooling layer supporting max and average pooling. The layer keeps track of
 * the locations of maxima for backprop when operating in max mode.
 */
export class PoolLayer {
  type = LayerType.POOL;
  size: number;
  stride: number;
  padding: number;
  op: "max" | "avg";

  input: number[][][] = null;
  output: number[][][] = null;
  private maxPositions: {y: number, x: number}[][][] = null;

  constructor(config: PoolLayerConfig) {
    this.size = config.size;
    this.stride = config.stride || config.size;
    this.padding = config.padding || 0;
    this.op = config.op;
  }

  forward(input: number[][][]): number[][][] {
    this.input = input;
    const inH = input.length;
    const inW = input[0].length;
    const depth = input[0][0].length;
    const outH = Math.floor((inH - this.size + 2 * this.padding) /
                                this.stride) + 1;
    const outW = Math.floor((inW - this.size + 2 * this.padding) /
                                this.stride) + 1;
    this.output = zeros3D(outH, outW, depth);
    if (this.op === "max") {
      this.maxPositions = new Array(outH);
      for (let y = 0; y < outH; y++) {
        this.maxPositions[y] = new Array(outW);
        for (let x = 0; x < outW; x++) {
          this.maxPositions[y][x] = new Array(depth);
        }
      }
    }
    for (let d = 0; d < depth; d++) {
      for (let y = 0; y < outH; y++) {
        for (let x = 0; x < outW; x++) {
          let best = this.op === "max" ? -Infinity : 0;
          let count = 0;
          let maxPos = {y: 0, x: 0};
          for (let ky = 0; ky < this.size; ky++) {
            for (let kx = 0; kx < this.size; kx++) {
              const inY = y * this.stride + ky - this.padding;
              const inX = x * this.stride + kx - this.padding;
              if (inY < 0 || inY >= inH || inX < 0 || inX >= inW) {
                continue;
              }
              const val = input[inY][inX][d];
              if (this.op === "max") {
                if (val > best) {
                  best = val;
                  maxPos = {y: inY, x: inX};
                }
              } else {
                best += val;
                count++;
              }
            }
          }
          if (this.op === "max") {
            this.output[y][x][d] = best;
            this.maxPositions[y][x][d] = maxPos;
          } else {
            this.output[y][x][d] = count === 0 ? 0 : best / count;
          }
        }
      }
    }
    return this.output;
  }

  backward(grad: number[][][]): number[][][] {
    const inH = this.input.length;
    const inW = this.input[0].length;
    const depth = this.input[0][0].length;
    const outH = grad.length;
    const outW = grad[0].length;
    const gradInput = zeros3D(inH, inW, depth);
    for (let d = 0; d < depth; d++) {
      for (let y = 0; y < outH; y++) {
        for (let x = 0; x < outW; x++) {
          const g = grad[y][x][d];
          if (this.op === "max") {
            const pos = this.maxPositions[y][x][d];
            if (pos) {
              gradInput[pos.y][pos.x][d] += g;
            }
          } else {
            for (let ky = 0; ky < this.size; ky++) {
              for (let kx = 0; kx < this.size; kx++) {
                const inY = y * this.stride + ky - this.padding;
                const inX = x * this.stride + kx - this.padding;
                if (inY < 0 || inY >= inH || inX < 0 || inX >= inW) {
                  continue;
                }
                gradInput[inY][inX][d] +=
                    g / (this.size * this.size);
              }
            }
          }
        }
      }
    }
    return gradInput;
  }

  getOutputShape(): [number, number, number] {
    return [this.output.length, this.output[0].length,
      this.input[0][0].length];
  }

  updateWeights() {
    // Pooling layers have no weights.
  }
}

// Utility helpers ---------------------------------------------------------

function zeros3D(h: number, w: number, d: number): number[][][] {
  let res: number[][][] = new Array(h);
  for (let y = 0; y < h; y++) {
    res[y] = new Array(w);
    for (let x = 0; x < w; x++) {
      res[y][x] = [];
      for (let z = 0; z < d; z++) {
        res[y][x][z] = 0;
      }
    }
  }
  return res;
}

function flatten3D(t: number[][][]): number[] {
  let res: number[] = [];
  for (let y = 0; y < t.length; y++) {
    for (let x = 0; x < t[0].length; x++) {
      for (let d = 0; d < t[0][0].length; d++) {
        res.push(t[y][x][d]);
      }
    }
  }
  return res;
}

function reshape1DTo3D(arr: number[], shape: [number, number, number]): number[][][] {
  const [h, w, d] = shape;
  const res = zeros3D(h, w, d);
  let idx = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      for (let z = 0; z < d; z++) {
        res[y][x][z] = arr[idx++];
      }
    }
  }
  return res;
}

/**
 * Builds a neural network.
 *
 * @param networkShape The shape of the network. E.g. [1, 2, 3, 1] means
 *   the network will have one input node, 2 nodes in first hidden layer,
 *   3 nodes in second hidden layer and 1 output node.
 * @param activation The activation function of every hidden node.
 * @param outputActivation The activation function for the output nodes.
 * @param regularization The regularization function that computes a penalty
 *     for a given weight (parameter) in the network. If null, there will be
 *     no regularization.
 * @param inputIds List of ids for the input nodes.
 */
export function buildNetwork(
    networkShape: Array<number | ConvLayerConfig | PoolLayerConfig>,
    activation: ActivationFunction,
    outputActivation: ActivationFunction,
    regularization: RegularizationFunction,
    inputIds: string[], initZero?: boolean): any[] {
  // If the networkShape is specified as numbers we build a classic dense
  // network. Otherwise we interpret the entries as layer configuration objects
  // and build a network consisting of convolutional and/or pooling layers.
  if (networkShape.length === 0) {
    return [];
  }
  if (typeof networkShape[0] !== "number") {
    let network: any[] = [];
    for (let i = 0; i < networkShape.length; i++) {
      let spec: any = networkShape[i];
      if (spec.type === LayerType.CONV) {
        network.push(new ConvolutionLayer(spec as ConvLayerConfig,
            regularization));
      } else if (spec.type === LayerType.POOL) {
        network.push(new PoolLayer(spec as PoolLayerConfig));
      }
    }
    return network;
  }

  let numLayers = networkShape.length;
  let id = 1;
  let network: Node[][] = [];
  for (let layerIdx = 0; layerIdx < numLayers; layerIdx++) {
    let isOutputLayer = layerIdx === numLayers - 1;
    let isInputLayer = layerIdx === 0;
    let currentLayer: Node[] = [];
    network.push(currentLayer);
    let numNodes = networkShape[layerIdx] as number;
    for (let i = 0; i < numNodes; i++) {
      let nodeId = id.toString();
      if (isInputLayer) {
        nodeId = inputIds[i];
      } else {
        id++;
      }
      let node = new Node(nodeId,
          isOutputLayer ? outputActivation : activation, initZero);
      currentLayer.push(node);
      if (layerIdx >= 1) {
        // Add links from nodes in the previous layer to this node.
        for (let j = 0; j < network[layerIdx - 1].length; j++) {
          let prevNode = network[layerIdx - 1][j];
          let link = new Link(prevNode, node, regularization, initZero);
          prevNode.outputs.push(link);
          node.inputLinks.push(link);
        }
      }
    }
  }
  return network;
}

/**
 * Runs a forward propagation of the provided input through the provided
 * network. This method modifies the internal state of the network - the
 * total input and output of each node in the network.
 *
 * @param network The neural network.
 * @param inputs The input array. Its length should match the number of input
 *     nodes in the network.
 * @return The final output of the network.
 */
export function forwardProp(network: any[], inputs: any): number {
  if (network.length === 0) {
    return 0;
  }

  // Dense network path.
  if (Array.isArray(network[0])) {
    let denseNet = network as Node[][];
    let inputLayer = denseNet[0];
    if (inputs.length !== inputLayer.length) {
      throw new Error("The number of inputs must match the number of nodes in" +
          " the input layer");
    }
    // Update the input layer.
    for (let i = 0; i < inputLayer.length; i++) {
      let node = inputLayer[i];
      node.output = inputs[i];
    }
    for (let layerIdx = 1; layerIdx < denseNet.length; layerIdx++) {
      let currentLayer = denseNet[layerIdx];
      for (let i = 0; i < currentLayer.length; i++) {
        currentLayer[i].updateOutput();
      }
    }
    return denseNet[denseNet.length - 1][0].output;
  }

  // Convolutional / pooling network path.
  let activation = inputs;
  for (let i = 0; i < network.length; i++) {
    let layer = network[i];
    if (layer instanceof ConvolutionLayer) {
      activation = layer.forward(activation);
    } else if (layer instanceof PoolLayer) {
      activation = layer.forward(activation);
    }
  }
  // Assume the network outputs a single scalar value in a 1x1x1 tensor.
  return activation[0][0][0];
}

/**
 * Runs a backward propagation using the provided target and the
 * computed output of the previous call to forward propagation.
 * This method modifies the internal state of the network - the error
 * derivatives with respect to each node, and each weight
 * in the network.
 */
export function backProp(network: any[], target: number,
    errorFunc: ErrorFunction): void {
  if (network.length === 0) {
    return;
  }

  // Dense network path.
  if (Array.isArray(network[0])) {
    let denseNet = network as Node[][];
    let outputNode = denseNet[denseNet.length - 1][0];
    outputNode.outputDer = errorFunc.der(outputNode.output, target);

    for (let layerIdx = denseNet.length - 1; layerIdx >= 1; layerIdx--) {
      let currentLayer = denseNet[layerIdx];
      for (let i = 0; i < currentLayer.length; i++) {
        let node = currentLayer[i];
        node.inputDer = node.outputDer * node.activation.der(node.totalInput);
        node.accInputDer += node.inputDer;
        node.numAccumulatedDers++;
      }
      for (let i = 0; i < currentLayer.length; i++) {
        let node = currentLayer[i];
        for (let j = 0; j < node.inputLinks.length; j++) {
          let link = node.inputLinks[j];
          if (link.isDead) {
            continue;
          }
          link.errorDer = node.inputDer * link.source.output;
          link.accErrorDer += link.errorDer;
          link.numAccumulatedDers++;
        }
      }
      if (layerIdx === 1) {
        continue;
      }
      let prevLayer = denseNet[layerIdx - 1];
      for (let i = 0; i < prevLayer.length; i++) {
        let node = prevLayer[i];
        node.outputDer = 0;
        for (let j = 0; j < node.outputs.length; j++) {
          let output = node.outputs[j];
          node.outputDer += output.weight * output.dest.inputDer;
        }
      }
    }
    return;
  }

  // Convolutional / pooling network path.
  let lastLayer: any = network[network.length - 1];
  // Assume scalar output at [0][0][0].
  let outputVal = lastLayer.output[0][0][0];
  let grad: number[][][] = [[[errorFunc.der(outputVal, target)]]];
  for (let layerIdx = network.length - 1; layerIdx >= 0; layerIdx--) {
    let layer = network[layerIdx];
    grad = layer.backward(grad);
  }
}

/**
 * Updates the weights of the network using the previously accumulated error
 * derivatives.
 */
export function updateWeights(network: any[], learningRate: number,
    regularizationRate: number) {
  if (network.length === 0) {
    return;
  }

  // Dense network path.
  if (Array.isArray(network[0])) {
    let denseNet = network as Node[][];
    for (let layerIdx = 1; layerIdx < denseNet.length; layerIdx++) {
      let currentLayer = denseNet[layerIdx];
      for (let i = 0; i < currentLayer.length; i++) {
        let node = currentLayer[i];
        if (node.numAccumulatedDers > 0) {
          node.bias -= learningRate * node.accInputDer / node.numAccumulatedDers;
          node.accInputDer = 0;
          node.numAccumulatedDers = 0;
        }
        for (let j = 0; j < node.inputLinks.length; j++) {
          let link = node.inputLinks[j];
          if (link.isDead) {
            continue;
          }
          let regulDer = link.regularization ?
              link.regularization.der(link.weight) : 0;
          if (link.numAccumulatedDers > 0) {
            link.weight = link.weight -
                (learningRate / link.numAccumulatedDers) * link.accErrorDer;
            let newLinkWeight = link.weight -
                (learningRate * regularizationRate) * regulDer;
            if (link.regularization === RegularizationFunction.L1 &&
                link.weight * newLinkWeight < 0) {
              link.weight = 0;
              link.isDead = true;
            } else {
              link.weight = newLinkWeight;
            }
            link.accErrorDer = 0;
            link.numAccumulatedDers = 0;
          }
        }
      }
    }
    return;
  }

  // Convolutional / pooling network path.
  for (let i = 0; i < network.length; i++) {
    let layer = network[i];
    if (layer instanceof ConvolutionLayer) {
      layer.updateWeights(learningRate, regularizationRate);
    }
    // Pooling layers have no weights to update.
  }
}

/** Iterates over every node in the network/ */
export function forEachNode(network: Node[][], ignoreInputs: boolean,
    accessor: (node: Node) => any) {
  for (let layerIdx = ignoreInputs ? 1 : 0;
      layerIdx < network.length;
      layerIdx++) {
    let currentLayer = network[layerIdx];
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      accessor(node);
    }
  }
}

/** Returns the output node in the network. */
export function getOutputNode(network: Node[][]) {
  return network[network.length - 1][0];
}
