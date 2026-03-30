const STATUS_COLORS = {
  pending: "#d1d5db",
  running: "#60a5fa",
  completed: "#86efac",
  failed: "#f87171",
};

function createSvgElement(name) {
  return document.createElementNS("http://www.w3.org/2000/svg", name);
}

function getNodeStroke(typeName) {
  if (typeName === "split-switch") {
    return { color: "#7c3aed", dash: "2 2", width: "2" };
  }
  if (typeName === "split-foreach" || typeName === "split-parallel") {
    return { color: "#0f766e", dash: "5 3", width: "2" };
  }
  if (typeName === "split-static") {
    return { color: "#1d4ed8", dash: "", width: "2" };
  }
  return { color: "#374151", dash: "", width: "1.5" };
}

function drawEdge(svg, markerId, sourceNode, targetNode, edge) {
  const sourceX = sourceNode.x + sourceNode.width;
  const sourceY = sourceNode.y + sourceNode.height / 2;
  const targetX = targetNode.x;
  const targetY = targetNode.y + targetNode.height / 2;
  const midX = (sourceX + targetX) / 2;

  const path = createSvgElement("path");
  path.setAttribute(
    "d",
    `M ${sourceX} ${sourceY} C ${midX} ${sourceY}, ${midX} ${targetY}, ${targetX} ${targetY}`
  );
  path.setAttribute("fill", "none");
  path.setAttribute("stroke", "#6b7280");
  path.setAttribute("stroke-width", "1.5");
  path.setAttribute("marker-end", `url(#${markerId})`);
  svg.appendChild(path);

  if (edge.label) {
    const label = createSvgElement("text");
    label.setAttribute("x", String(midX));
    label.setAttribute("y", String((sourceY + targetY) / 2 - 4));
    label.setAttribute("font-size", "10");
    label.setAttribute("fill", "#4b5563");
    label.setAttribute("text-anchor", "middle");
    label.textContent = edge.label;
    svg.appendChild(label);
  }
}

function drawNode(svg, model, node, stepStatus, selectedStep, showRunDetails) {
  const statusName =
    stepStatus && typeof stepStatus.status === "string"
      ? stepStatus.status
      : "pending";
  const fillColor = STATUS_COLORS[statusName] || STATUS_COLORS.pending;
  const stroke = getNodeStroke(node.type);

  const group = createSvgElement("g");
  group.setAttribute("transform", `translate(${node.x}, ${node.y})`);
  group.style.cursor = "pointer";

  if (node.type === "split-switch") {
    const polygon = createSvgElement("polygon");
    const w = node.width;
    const h = node.height;
    polygon.setAttribute(
      "points",
      `${w * 0.16},0 ${w * 0.84},0 ${w},${h / 2} ${w * 0.84},${h} ${w * 0.16},${h} 0,${h / 2}`
    );
    polygon.setAttribute("fill", fillColor);
    polygon.setAttribute(
      "stroke",
      selectedStep === node.id ? "#111827" : stroke.color
    );
    polygon.setAttribute(
      "stroke-width",
      selectedStep === node.id ? "3" : stroke.width
    );
    if (stroke.dash) {
      polygon.setAttribute("stroke-dasharray", stroke.dash);
    }
    group.appendChild(polygon);
  } else {
    const rect = createSvgElement("rect");
    rect.setAttribute("width", String(node.width));
    rect.setAttribute("height", String(node.height));
    rect.setAttribute("rx", "10");
    rect.setAttribute("ry", "10");
    rect.setAttribute("fill", fillColor);
    rect.setAttribute("stroke", selectedStep === node.id ? "#111827" : stroke.color);
    rect.setAttribute(
      "stroke-width",
      selectedStep === node.id ? "3" : stroke.width
    );
    if (stroke.dash) {
      rect.setAttribute("stroke-dasharray", stroke.dash);
    }
    group.appendChild(rect);
  }

  const title = createSvgElement("text");
  title.setAttribute("x", "10");
  title.setAttribute("y", showRunDetails ? "17" : "25");
  title.setAttribute("font-size", "12");
  title.setAttribute("font-weight", "600");
  title.setAttribute("fill", "#111827");
  title.textContent = node.label;
  group.appendChild(title);

  if (showRunDetails) {
    const subtitle = createSvgElement("text");
    subtitle.setAttribute("x", "10");
    subtitle.setAttribute("y", "33");
    subtitle.setAttribute("font-size", "10");
    subtitle.setAttribute("fill", "#374151");
    subtitle.textContent = `${node.type || "unknown"} | ${statusName}`;
    group.appendChild(subtitle);
  }

  group.addEventListener("click", () => {
    model.send({ type: "select_step", step: node.id });
  });

  svg.appendChild(group);
}

function renderGraph(model, container, svg) {
  while (svg.firstChild) {
    svg.removeChild(svg.firstChild);
  }

  const graph = model.get("graph_model") || {};
  const nodes = Array.isArray(graph.nodes) ? graph.nodes : [];
  const edges = Array.isArray(graph.edges) ? graph.edges : [];
  const meta = graph.meta || {};
  const statuses = model.get("step_statuses") || {};
  const selectedStep = model.get("selected_step");

  const width = Math.max(320, Number(meta.width || 0));
  const height = Math.max(120, Number(meta.height || 0));

  svg.setAttribute("width", String(width));
  svg.setAttribute("height", String(height));
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);

  if (nodes.length === 0) {
    const waiting = createSvgElement("text");
    waiting.setAttribute("x", String(width / 2));
    waiting.setAttribute("y", String(height / 2));
    waiting.setAttribute("font-size", "12");
    waiting.setAttribute("fill", "#6b7280");
    waiting.setAttribute("text-anchor", "middle");
    waiting.textContent = "Waiting for DAG metadata...";
    svg.appendChild(waiting);
    return;
  }

  const defs = createSvgElement("defs");
  const marker = createSvgElement("marker");
  marker.setAttribute("id", "mf-dag-arrow");
  marker.setAttribute("markerWidth", "8");
  marker.setAttribute("markerHeight", "8");
  marker.setAttribute("refX", "7");
  marker.setAttribute("refY", "4");
  marker.setAttribute("orient", "auto");
  const arrowPath = createSvgElement("path");
  arrowPath.setAttribute("d", "M 0 0 L 8 4 L 0 8 z");
  arrowPath.setAttribute("fill", "#6b7280");
  marker.appendChild(arrowPath);
  defs.appendChild(marker);
  svg.appendChild(defs);

  const nodeById = {};
  nodes.forEach((node) => {
    nodeById[node.id] = node;
  });

  edges.forEach((edge) => {
    const sourceNode = nodeById[edge.source];
    const targetNode = nodeById[edge.target];
    if (!sourceNode || !targetNode) {
      return;
    }
    drawEdge(svg, "mf-dag-arrow", sourceNode, targetNode, edge);
  });

  const showRunDetails = Boolean(meta.run_pathspec);

  nodes.forEach((node) => {
    drawNode(svg, model, node, statuses[node.id], selectedStep, showRunDetails);
  });

  const title = model.get("graph_model")?.meta?.run_pathspec
    ? `${model.get("graph_model").meta.flow_name} (${model.get("graph_model").meta.run_pathspec})`
    : `${model.get("graph_model")?.meta?.flow_name || "Flow"} DAG`;
  container.setAttribute("aria-label", title);
}

export default {
  render({ model, el }) {
    const wrapper = document.createElement("div");
    wrapper.className = "mf-dag-wrapper";
    const svg = createSvgElement("svg");
    svg.classList.add("mf-dag-svg");
    wrapper.appendChild(svg);
    el.appendChild(wrapper);

    let pollInterval = null;

    function updatePolling() {
      if (pollInterval !== null) {
        clearInterval(pollInterval);
        pollInterval = null;
      }
      const isLive = Boolean(model.get("live"));
      const intervalMs = Math.max(
        200,
        Number(model.get("poll_interval_ms") || 2000)
      );
      if (isLive) {
        // Poll once immediately so first render doesn't stay stale for a full interval.
        model.send({ type: "poll" });
        pollInterval = setInterval(() => {
          model.send({ type: "poll" });
        }, intervalMs);
      }
    }

    function rerender() {
      renderGraph(model, wrapper, svg);
    }

    model.on("change:graph_model", rerender);
    model.on("change:step_statuses", rerender);
    model.on("change:selected_step", rerender);
    model.on("change:live", updatePolling);
    model.on("change:poll_interval_ms", updatePolling);

    rerender();
    updatePolling();
    model.send({ type: "request_graph_sync" });

    return () => {
      if (pollInterval !== null) {
        clearInterval(pollInterval);
      }
      model.off("change:graph_model", rerender);
      model.off("change:step_statuses", rerender);
      model.off("change:selected_step", rerender);
      model.off("change:live", updatePolling);
      model.off("change:poll_interval_ms", updatePolling);
      el.innerHTML = "";
    };
  },
};
