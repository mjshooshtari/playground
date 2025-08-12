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

import * as d3 from 'd3';

export interface FeatureMapData {
  data: number[][];
  label?: string;
}

export class FeatureMapPanel {
  private root: d3.Selection<any>;
  private grid: d3.Selection<any>;
  private scale: number;

  constructor(container: d3.Selection<any>, scale = 4) {
    this.scale = scale;
    this.root = container.append('div')
      .attr('class', 'feature-map-panel');
    this.grid = this.root.append('div')
      .attr('class', 'feature-map-grid');
  }

  update(maps: number[][][], labels?: string[]): void {
    let data: FeatureMapData[] = maps.map((m, i) => {
      return {data: m, label: labels ? labels[i] : null};
    });

    let selection = this.grid.selectAll('div.feature-map').data(data);

    let enter = selection.enter().append('div')
      .attr('class', 'feature-map');
    enter.append('canvas');
    enter.append('div').attr('class', 'caption');

    selection.exit().remove();

    selection.select('canvas').each((d: FeatureMapData, i: number, nodes) => {
      this.renderMatrix(nodes[i] as HTMLCanvasElement, d.data);
    });

    selection.select('div.caption')
      .text((d: FeatureMapData) => d.label || '');
  }

  private renderMatrix(canvas: HTMLCanvasElement, matrix: number[][]): void {
    let height = matrix.length;
    let width = matrix[0].length;

    canvas.width = width;
    canvas.height = height;
    canvas.style.width = `${width * this.scale}px`;
    canvas.style.height = `${height * this.scale}px`;

    let ctx = canvas.getContext('2d');
    let image = ctx.createImageData(width, height);

    let min = Infinity;
    let max = -Infinity;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let v = matrix[y][x];
        if (v < min) min = v;
        if (v > max) max = v;
      }
    }
    let range = max - min || 1;

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let v = matrix[y][x];
        let norm = Math.round(255 * (v - min) / range);
        let idx = (y * width + x) * 4;
        image.data[idx] = norm;
        image.data[idx + 1] = norm;
        image.data[idx + 2] = norm;
        image.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(image, 0, 0);
  }
}

