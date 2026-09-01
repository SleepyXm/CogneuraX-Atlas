// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { CustomComponent } from "./types.js";

function actionFromClass<N, P>(
  ComponentClass: new (
    node: N,
    props: P,
  ) => {
    update?: ((props: P) => void) | undefined;
    destroy?: () => void;
  },
) {
  return (node: N, props: P & Record<string, any>) => {
    let obj = new ComponentClass(node, props);
    return {
      ...(obj.update ? { update: obj.update.bind(obj) } : {}),
      ...(obj.destroy ? { destroy: obj.destroy.bind(obj) } : {}),
    };
  };
}

let componentActionCache = new WeakMap();

export function customComponentAction<N, P>(component: CustomComponent<N, P>) {
  let ComponentClass = typeof component == "function" ? component : component.class;
  if (componentActionCache.has(ComponentClass)) {
    return componentActionCache.get(ComponentClass);
  } else {
    let action = actionFromClass(ComponentClass);
    componentActionCache.set(ComponentClass, action);
    return action;
  }
}

export function customComponentProps<N, P>(component: CustomComponent<N, P>, props: P) {
  if (typeof component == "function") {
    return props;
  } else {
    return { ...(component.props ?? {}), ...props };
  }
}
