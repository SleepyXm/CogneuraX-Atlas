<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts" module>
  interface Props<Tooltip> {
    /** Location in pixels */
    location: Point;
    /** Height of the point target */
    targetHeight: number;
    /** Whether to allow pointer interaction */
    allowInteraction: boolean;

    tooltip: Tooltip;
    customTooltip: CustomComponent<HTMLDivElement, { tooltip: Tooltip }>;
    /** Y margin */
    margin?: number;
  }
</script>

<script lang="ts">
  import { onMount } from "svelte";

  import type { Point } from "../utils.js";
  import { customComponentAction, customComponentProps } from "./custom_component_helper.js";
  import type { CustomComponent } from "./types.js";

  type Tooltip = $$Generic;

  let { location, targetHeight, allowInteraction, tooltip, customTooltip, margin = 4 }: Props<Tooltip> = $props();

  let parentContainer: HTMLDivElement;
  let container: HTMLDivElement;

  let action = $derived(customComponentAction(customTooltip));
  let instanceProps = $derived(customComponentProps(customTooltip, { tooltip: tooltip }));

  onMount(() => {
    $effect.pre(() => {
      let capturedAction = action;
      let instance: any | null = null;

      $effect.pre(() => {
        container.style.left = "0px";
        container.style.top = "0px";
        container.style.pointerEvents = allowInteraction ? "all" : "none";

        if (instance == null) {
          instance = capturedAction(container, instanceProps);
        } else {
          instance.update?.(instanceProps);
        }

        function updatePosition(width: number, height: number, xMin: number, xMax: number) {
          let px = location.x;
          let py = location.y;
          let yMin = 2;
          let anchorX = width / 2;
          let anchorY = height + (targetHeight + margin);
          if (px - anchorX < xMin) {
            anchorX = px - xMin;
          }
          if (px - anchorX > xMax - width) {
            anchorX = px - xMax + width;
          }
          if (py - anchorY < yMin) {
            anchorY = -(targetHeight + margin);
          }
          container.style.left = px - anchorX + "px";
          container.style.top = py - anchorY + "px";
        }

        let parentRect = parentContainer.getBoundingClientRect();
        let { width, height } = container.getBoundingClientRect();
        updatePosition(width, height, 2, parentRect.width - 2);

        // Sometimes the size may change in the next animation frame, add this to double check.
        let req: number | null = requestAnimationFrame(() => {
          req = null;

          let rect = container.getBoundingClientRect();
          if (rect.width != width || rect.height != height) {
            updatePosition(rect.width, rect.height, 2, parentRect.width - 2);
          }
        });
        return () => {
          if (req != null) {
            cancelAnimationFrame(req);
          }
        };
      });

      return () => {
        instance?.destroy?.();
        container.replaceChildren();
      };
    });
  });
</script>

<div bind:this={parentContainer} style:position="absolute" style:width="100%">
  <div
    bind:this={container}
    style:display="flex"
    style:position="absolute"
    style:width="fit-content"
    style:height="fit-content"
    style:z-index="100"
  ></div>
</div>
