<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  interface Props {
    distancePerPoint: number;
  }

  let { distancePerPoint }: Props = $props();

  const desiredLength = 30;

  function findLabelUnit(distancePerPoint: number, desiredLength: number) {
    let log10t = Math.log10(desiredLength * distancePerPoint);
    let x = Math.round(log10t);
    let offsets = [0.1, 0.2, 0.5, 1, 2, 5, 10];
    let minOffset = 0;
    let minDiff = 1e10;
    for (let d of offsets) {
      let diff = Math.abs(Math.log10(d) + x - log10t);
      if (diff < minDiff) {
        minOffset = d;
        minDiff = diff;
      }
    }
    return minOffset * Math.pow(10, x);
  }

  let labelUnit = $derived(findLabelUnit(distancePerPoint, desiredLength));
  let length = $derived(labelUnit / distancePerPoint);
</script>

<div style:display="flex" style:align-items="center">
  <div style:padding-right="4px">{labelUnit.toLocaleString()}</div>
  <svg width="{length + 2}px" height="6px">
    <line
      x1={1}
      x2={length + 1}
      y1={3}
      y2={3}
      style:stroke="currentColor"
      style:stroke-width="2"
      style:stroke-cap="butt"
      shape-rendering="crispEdges"
    />
    <line x1={1} x2={1} y1={0} y2={6} style:stroke="currentColor" shape-rendering="crispEdges" />
    <line x1={length + 1} x2={length + 1} y1={0} y2={6} style:stroke="currentColor" shape-rendering="crispEdges" />
  </svg>
</div>
