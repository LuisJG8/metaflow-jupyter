import heapq
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from metaflow import FlowSpec, Run

try:
    import anywidget
    import traitlets
except ImportError:
    anywidget = None
    traitlets = None


_ANYWIDGET_AVAILABLE = anywidget is not None and traitlets is not None
_MISSING_ANYWIDGET_ERROR = (
    "DagWidget requires optional dependency 'anywidget'. "
    "Install with: pip install anywidget"
)

_STATUS_PENDING = "pending"
_STATUS_RUNNING = "running"
_STATUS_COMPLETED = "completed"
_STATUS_FAILED = "failed"

_VALID_STATUS_VALUES = {
    _STATUS_PENDING,
    _STATUS_RUNNING,
    _STATUS_COMPLETED,
    _STATUS_FAILED,
}


def _is_flowspec_subclass(source: Any) -> bool:
    return isinstance(source, type) and issubclass(source, FlowSpec)


def _is_executing_run_instance(source: Any) -> bool:
    try:
        from .metaflow_runner import ExecutingRun
    except Exception:
        return False
    return isinstance(source, ExecutingRun)


def _coerce_run_from_source(source: Any) -> Optional[Run]:
    if isinstance(source, Run):
        return source

    if _is_executing_run_instance(source):
        run_obj = getattr(source, "run", None)
        if isinstance(run_obj, Run):
            return run_obj
        return None

    try:
        run_obj = getattr(source, "run")
    except Exception:
        return None
    return run_obj if isinstance(run_obj, Run) else None


def _normalize_steps(steps: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    normalized = {}
    for step_name, spec in (steps or {}).items():
        if spec is None:
            spec = {}
        next_steps = spec.get("next") or []
        normalized[step_name] = {
            "type": spec.get("type") or "unknown",
            "doc": spec.get("doc") or "",
            "next": [str(target) for target in next_steps],
            "matching_join": spec.get("matching_join"),
            "switch_cases": deepcopy(spec.get("switch_cases") or {}),
        }

    # Add placeholders for referenced nodes that are not declared in the payload.
    for spec in list(normalized.values()):
        for target in spec["next"]:
            if target not in normalized:
                normalized[target] = {
                    "type": "unknown",
                    "doc": "",
                    "next": [],
                    "matching_join": None,
                    "switch_cases": {},
                }

    return normalized


def _extract_steps_from_flowspec(flow_cls: Any) -> Dict[str, Dict[str, Any]]:
    if not _is_flowspec_subclass(flow_cls):
        raise TypeError("Expected a FlowSpec subclass.")

    if getattr(flow_cls, "_graph", None) is None:
        flow_cls._init_graph()
    steps_info, _graph_structure = flow_cls._graph.output_steps()
    return _normalize_steps(steps_info)


def _extract_graph_info_from_run(run_obj: Any) -> Dict[str, Any]:
    # The _parameters task should usually contain _graph_info very early.
    candidates = []
    try:
        parameters_step = run_obj["_parameters"]
        if parameters_step and parameters_step.task is not None:
            candidates.append(parameters_step.task)
    except Exception:
        pass

    # Fallback: scan available tasks from visible steps.
    try:
        for step_obj in run_obj:
            task_obj = step_obj.task
            if task_obj is not None:
                candidates.append(task_obj)
    except Exception:
        pass

    seen_pathspecs = set()
    for task_obj in candidates:
        task_pathspec = getattr(task_obj, "pathspec", id(task_obj))
        if task_pathspec in seen_pathspecs:
            continue
        seen_pathspecs.add(task_pathspec)

        try:
            graph_info = task_obj["_graph_info"].data
        except Exception:
            continue
        if isinstance(graph_info, dict) and isinstance(graph_info.get("steps"), dict):
            return graph_info

    raise ValueError(
        "Unable to resolve flow graph metadata for run '%s'."
        % getattr(run_obj, "pathspec", "<unknown>")
    )


def _extract_steps_from_run(run_obj: Any) -> Dict[str, Dict[str, Any]]:
    graph_info = _extract_graph_info_from_run(run_obj)
    return _normalize_steps(graph_info.get("steps") or {})


def _normalize_source(
    source: Any,
) -> Tuple[str, str, Optional[Any], Dict[str, Dict[str, Any]]]:
    if _is_flowspec_subclass(source):
        steps = _extract_steps_from_flowspec(source)
        return "static", source.__name__, None, steps

    run_obj = _coerce_run_from_source(source)
    if run_obj is not None:
        steps = _extract_steps_from_run(run_obj)
        flow_name = run_obj.path_components[0] if run_obj.path_components else "<unknown>"
        return "live", flow_name, run_obj, steps

    raise TypeError(
        "DagWidget source must be a FlowSpec subclass, Run, or ExecutingRun."
    )


def _build_edges(steps: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    edges = []
    for step_name, spec in steps.items():
        switch_cases = spec.get("switch_cases") or {}
        labels_by_target = defaultdict(list)
        for case_key, case_target in switch_cases.items():
            labels_by_target[case_target].append(str(case_key))

        for target_name in spec.get("next") or []:
            labels = labels_by_target.get(target_name) or []
            edge_label = ", ".join(labels) if labels else None
            edge_kind = "switch" if labels else "normal"
            edges.append(
                {
                    "source": step_name,
                    "target": target_name,
                    "label": edge_label,
                    "kind": edge_kind,
                }
            )
    return edges


def _compute_topological_order(
    nodes: Iterable[str], adjacency: Dict[str, List[str]]
) -> List[str]:
    indegree = {node: 0 for node in nodes}
    for src in adjacency:
        for dst in adjacency[src]:
            indegree[dst] = indegree.get(dst, 0) + 1

    heap = []
    for node_name, deg in indegree.items():
        if deg == 0:
            heapq.heappush(heap, node_name)

    topo = []
    while heap:
        node_name = heapq.heappop(heap)
        topo.append(node_name)
        for dst in sorted(adjacency.get(node_name, [])):
            indegree[dst] -= 1
            if indegree[dst] == 0:
                heapq.heappush(heap, dst)

    # If cycles exist, append unresolved nodes deterministically.
    if len(topo) < len(indegree):
        unresolved = sorted(node for node in indegree if node not in topo)
        topo.extend(unresolved)

    return topo


def _compute_layout(
    steps: Dict[str, Dict[str, Any]],
    node_width: int,
    node_height: int,
    h_gap: int,
    v_gap: int,
) -> Dict[str, Dict[str, float]]:
    adjacency = defaultdict(list)
    parents = defaultdict(list)
    for step_name, spec in steps.items():
        for target_name in spec.get("next") or []:
            adjacency[step_name].append(target_name)
            parents[target_name].append(step_name)
        adjacency.setdefault(step_name, [])

    topo = _compute_topological_order(steps.keys(), adjacency)
    topo_index = {name: idx for idx, name in enumerate(topo)}

    layer = {name: 0 for name in steps}
    for step_name in topo:
        base_layer = layer.get(step_name, 0)
        for target_name in adjacency.get(step_name, []):
            candidate = base_layer + 1
            if candidate > layer.get(target_name, 0):
                layer[target_name] = candidate

    layers = defaultdict(list)
    for step_name in topo:
        layers[layer.get(step_name, 0)].append(step_name)

    spacing_y = float(node_height + v_gap)
    positions = {}
    order_in_layer = {}

    for layer_idx in sorted(layers):
        layer_nodes = list(layers[layer_idx])

        def _sort_key(step_name: str) -> Tuple[float, int, str]:
            parent_nodes = parents.get(step_name) or []
            if parent_nodes:
                barycenter = sum(
                    float(order_in_layer.get(parent, topo_index[parent]))
                    for parent in parent_nodes
                ) / float(len(parent_nodes))
            else:
                barycenter = float("inf")
            return (barycenter, topo_index[step_name], step_name)

        ordered = sorted(layer_nodes, key=_sort_key)
        proposed = []
        for idx, step_name in enumerate(ordered):
            parent_nodes = parents.get(step_name) or []
            if len(parent_nodes) > 1:
                parent_centers = []
                for parent_name in parent_nodes:
                    if parent_name in positions:
                        parent_y = positions[parent_name]["y"]
                        parent_centers.append(parent_y + (float(node_height) / 2.0))
                if parent_centers:
                    proposed_y = (
                        sum(parent_centers) / float(len(parent_centers))
                    ) - (float(node_height) / 2.0)
                else:
                    proposed_y = float(idx) * spacing_y
            else:
                proposed_y = float(idx) * spacing_y
            proposed.append((step_name, proposed_y))

        proposed.sort(key=lambda item: (item[1], topo_index[item[0]], item[0]))
        current_y = None
        for rank, (step_name, target_y) in enumerate(proposed):
            if current_y is None:
                current_y = max(0.0, target_y)
            else:
                current_y = max(target_y, current_y + spacing_y)

            positions[step_name] = {
                "x": float(layer_idx * (node_width + h_gap)),
                "y": float(current_y),
                "layer": int(layer_idx),
                "order": int(rank),
            }
            order_in_layer[step_name] = rank

    # Ensure every node has a position.
    for step_name in steps:
        positions.setdefault(
            step_name,
            {
                "x": 0.0,
                "y": 0.0,
                "layer": 0,
                "order": 0,
            },
        )

    return positions


def _build_graph_model(
    steps: Dict[str, Dict[str, Any]],
    flow_name: str,
    run_pathspec: Optional[str],
    node_width: int,
    node_height: int,
    h_gap: int,
    v_gap: int,
) -> Dict[str, Any]:
    normalized_steps = _normalize_steps(steps)
    positions = _compute_layout(normalized_steps, node_width, node_height, h_gap, v_gap)
    edges = _build_edges(normalized_steps)

    nodes = []
    for step_name, spec in normalized_steps.items():
        pos = positions[step_name]
        nodes.append(
            {
                "id": step_name,
                "label": step_name,
                "type": spec.get("type") or "unknown",
                "doc": spec.get("doc") or "",
                "next": list(spec.get("next") or []),
                "matching_join": spec.get("matching_join"),
                "x": pos["x"],
                "y": pos["y"],
                "width": float(node_width),
                "height": float(node_height),
                "layer": int(pos["layer"]),
                "order": int(pos["order"]),
            }
        )

    nodes.sort(key=lambda node: (node["layer"], node["order"], node["id"]))

    max_x = 0.0
    max_y = 0.0
    for node in nodes:
        max_x = max(max_x, node["x"] + node["width"])
        max_y = max(max_y, node["y"] + node["height"])

    return {
        "nodes": nodes,
        "edges": edges,
        "meta": {
            "flow_name": flow_name,
            "run_pathspec": run_pathspec,
            "orientation": "left-to-right",
            "width": max_x + 24.0,
            "height": max_y + 24.0,
        },
    }


def _pending_status_payload() -> Dict[str, Any]:
    return {
        "status": _STATUS_PENDING,
        "total_tasks": 0,
        "successful_tasks": 0,
        "failed_tasks": 0,
    }


def _compute_step_statuses(
    run_obj: Optional[Any],
    graph_nodes: Iterable[Dict[str, Any]],
    process_status: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    statuses = {}
    for node in graph_nodes:
        step_name = node.get("id")
        statuses[step_name] = _pending_status_payload()

    if run_obj is None:
        return statuses

    for step_name in list(statuses.keys()):
        try:
            step_obj = run_obj[step_name]
        except Exception:
            continue

        try:
            tasks = list(step_obj)
        except Exception:
            continue

        if not tasks:
            continue

        total_tasks = len(tasks)
        successful_tasks = sum(
            1 for task in tasks if bool(getattr(task, "finished", False)) and bool(getattr(task, "successful", False))
        )
        failed_tasks = sum(
            1 for task in tasks if bool(getattr(task, "finished", False)) and not bool(getattr(task, "successful", False))
        )
        all_finished = all(bool(getattr(task, "finished", False)) for task in tasks)

        if failed_tasks > 0:
            status = _STATUS_FAILED
        elif total_tasks > 0 and all_finished and successful_tasks == total_tasks:
            status = _STATUS_COMPLETED
        elif total_tasks > 0:
            status = _STATUS_RUNNING
        else:
            status = _STATUS_PENDING

        if status not in _VALID_STATUS_VALUES:
            status = _STATUS_PENDING

        statuses[step_name] = {
            "status": status,
            "total_tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "failed_tasks": failed_tasks,
        }

    if process_status == "running":
        has_running = any(
            payload.get("status") == _STATUS_RUNNING for payload in statuses.values()
        )
        has_failed = any(
            payload.get("status") == _STATUS_FAILED for payload in statuses.values()
        )
        if not has_running and not has_failed:
            completed_steps = {
                step_name
                for step_name, payload in statuses.items()
                if payload.get("status") == _STATUS_COMPLETED
            }

            node_order = {}
            parents_by_step = defaultdict(list)
            for node in graph_nodes:
                step_name = node.get("id")
                node_order[step_name] = (
                    int(node.get("layer", 0)),
                    int(node.get("order", 0)),
                    step_name,
                )
                for target_name in node.get("next") or []:
                    parents_by_step[target_name].append(step_name)
                parents_by_step.setdefault(step_name, parents_by_step.get(step_name, []))

            candidate_running = []
            for step_name, payload in statuses.items():
                if payload.get("status") != _STATUS_PENDING:
                    continue
                parent_steps = parents_by_step.get(step_name) or []
                if not parent_steps:
                    if not completed_steps:
                        candidate_running.append(step_name)
                    continue
                if all(parent_name in completed_steps for parent_name in parent_steps):
                    candidate_running.append(step_name)

            candidate_running.sort(
                key=lambda step_name: node_order.get(step_name, (0, 0, step_name))
            )
            for step_name in candidate_running:
                statuses[step_name] = {
                    "status": _STATUS_RUNNING,
                    "total_tasks": 0,
                    "successful_tasks": 0,
                    "failed_tasks": 0,
                }

    return statuses


def _default_live(mode: str, live: Optional[bool]) -> bool:
    if live is not None:
        return bool(live)
    return mode == "live"


def _graph_model_has_nodes(graph_model: Dict[str, Any]) -> bool:
    nodes = graph_model.get("nodes") if isinstance(graph_model, dict) else None
    return bool(nodes)


_ASSET_DIR = Path(__file__).resolve().parent


def _load_widget_asset(filename: str) -> str:
    return (_ASSET_DIR / filename).read_text(encoding="utf-8")


_ESM = _load_widget_asset("metaflow_jupyter_dag.js")
_CSS = _load_widget_asset("metaflow_jupyter_dag.css")


if _ANYWIDGET_AVAILABLE:

    class DagWidget(anywidget.AnyWidget):
        _esm = _ESM
        _css = _CSS

        graph_model = traitlets.Dict(default_value={}).tag(sync=True)
        step_statuses = traitlets.Dict(default_value={}).tag(sync=True)
        live = traitlets.Bool(default_value=False).tag(sync=True)
        poll_interval_ms = traitlets.Int(default_value=2000).tag(sync=True)
        selected_step = traitlets.Unicode(default_value=None, allow_none=True).tag(
            sync=True
        )

        def __init__(
            self,
            source,
            poll_interval_seconds=2.0,
            live=None,
            node_width=160,
            node_height=44,
            h_gap=64,
            v_gap=24,
        ):
            self._status_source = source
            mode, flow_name, run_obj, steps = _normalize_source(source)
            self._mode = mode
            self._run_obj = run_obj
            self._steps = steps
            self._flow_name = flow_name
            self._node_width = int(node_width)
            self._node_height = int(node_height)
            self._h_gap = int(h_gap)
            self._v_gap = int(v_gap)
            self._graph_sync_seq = 0

            graph_model = _build_graph_model(
                steps=steps,
                flow_name=flow_name,
                run_pathspec=getattr(run_obj, "pathspec", None),
                node_width=self._node_width,
                node_height=self._node_height,
                h_gap=self._h_gap,
                v_gap=self._v_gap,
            )
            initial_statuses = _compute_step_statuses(
                run_obj,
                graph_model["nodes"],
                process_status=self._process_status_hint(),
            )

            poll_interval_ms = max(int(float(poll_interval_seconds) * 1000.0), 200)
            is_live = _default_live(mode, live)
            if run_obj is not None and bool(getattr(run_obj, "finished", False)):
                is_live = False

            super(DagWidget, self).__init__()
            self.graph_model = graph_model
            self.step_statuses = initial_statuses
            self.live = is_live
            self.poll_interval_ms = poll_interval_ms
            self.selected_step = None
            self._node_ids = set(node["id"] for node in graph_model["nodes"])
            self._needs_graph_resync = is_live

            self.on_msg(self._handle_frontend_message)

        def _process_status_hint(self) -> Optional[str]:
            try:
                status = getattr(self._status_source, "status", None)
            except Exception:
                return None
            return status if isinstance(status, str) else None

        def _sync_graph_model(self):
            """
            Force a graph_model trait update so frontends that missed the initial
            payload can recover on the next poll tick.
            """
            synced_graph = deepcopy(self.graph_model)
            meta = dict(synced_graph.get("meta") or {})
            self._graph_sync_seq += 1
            meta["_sync_seq"] = self._graph_sync_seq
            synced_graph["meta"] = meta
            self.graph_model = synced_graph

        def _maybe_refresh_graph_from_run(self):
            if self._run_obj is None:
                return
            if _graph_model_has_nodes(self.graph_model):
                return

            try:
                refreshed_steps = _extract_steps_from_run(self._run_obj)
            except Exception:
                return
            if not refreshed_steps:
                return

            refreshed_graph_model = _build_graph_model(
                steps=refreshed_steps,
                flow_name=self._flow_name,
                run_pathspec=getattr(self._run_obj, "pathspec", None),
                node_width=self._node_width,
                node_height=self._node_height,
                h_gap=self._h_gap,
                v_gap=self._v_gap,
            )
            if not _graph_model_has_nodes(refreshed_graph_model):
                return

            self._steps = refreshed_steps
            self.graph_model = refreshed_graph_model
            self._node_ids = set(node["id"] for node in refreshed_graph_model["nodes"])

        def _handle_frontend_message(self, _widget, content, _buffers):
            if not isinstance(content, dict):
                return

            message_type = content.get("type")
            if message_type == "poll":
                if self._run_obj is None:
                    return
                self._maybe_refresh_graph_from_run()
                if self._needs_graph_resync:
                    self._sync_graph_model()
                    self._needs_graph_resync = False
                self.step_statuses = _compute_step_statuses(
                    self._run_obj,
                    self.graph_model.get("nodes", []),
                    process_status=self._process_status_hint(),
                )
                if bool(getattr(self._run_obj, "finished", False)):
                    self.live = False
            elif message_type == "request_graph_sync":
                self._sync_graph_model()
            elif message_type == "select_step":
                selected = content.get("step")
                if selected in self._node_ids:
                    self.selected_step = selected
                else:
                    self.selected_step = None
else:

    class DagWidget(object):
        def __init__(
            self,
            source,
            poll_interval_seconds=2.0,
            live=None,
            node_width=160,
            node_height=44,
            h_gap=64,
            v_gap=24,
        ):
            raise ImportError(_MISSING_ANYWIDGET_ERROR)
