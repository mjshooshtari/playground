/* Placeholder for feature map visualization. */

import * as d3 from 'd3';
import * as nn from './nn';

export class FeatureMap {
  constructor(private container: d3.Selection<any>) {}

  update(network: nn.Node[][]): void {
    // TODO: implement feature map rendering.
  }
}
