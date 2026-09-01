// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import simplify from "simplify-js";
import { boundingRect, type Point } from "../utils.js";

function chaikin(points: Point[], iterations: number): Point[] {
  let newPoints = points.slice();
  for (let iter = 0; iter < iterations; iter++) {
    const smoothed = [];
    const len = newPoints.length;
    for (let i = 0; i < len; i++) {
      const p0 = newPoints[i];
      const p1 = newPoints[(i + 1) % len];
      const Q = {
        x: 0.75 * p0.x + 0.25 * p1.x,
        y: 0.75 * p0.y + 0.25 * p1.y,
      };
      const R = {
        x: 0.25 * p0.x + 0.75 * p1.x,
        y: 0.25 * p0.y + 0.75 * p1.y,
      };
      smoothed.push(Q, R);
    }
    newPoints = smoothed;
  }
  return newPoints;
}

export function simplifyPolygon(points: Point[], desiredPoints: number): Point[] {
  const smoothed = chaikin(points, 5);

  const c = boundingRect(smoothed);
  let tolerance = Math.max(c.xMax - c.xMin, c.yMax - c.yMin) / 100;
  let simplified = simplify(smoothed, tolerance);
  let iterations = 0;
  while (simplified.length > desiredPoints && iterations < 20) {
    tolerance *= 1.1;
    iterations += 1;
    simplified = simplify(smoothed, tolerance);
  }
  return simplified;
}
