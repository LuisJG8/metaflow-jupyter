import json
import shlex
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from IPython.core.error import UsageError
from IPython.display import JSON, display

from metaflow import Flow, Metaflow, Run
from metaflow.exception import (
    MetaflowInvalidPathspec,
    MetaflowNamespaceMismatch,
    MetaflowNotFound,
)

_MF_SHOW_USAGE = """Usage:
%mf_show <Flow>/<run|latest>/<step> <artifact>
%mf_show latest <step>.<artifact>
%mf_show <step>.<artifact>"""


@dataclass(frozen=True)
class MFShowSpec:
    flow_name: Optional[str]
    run_id: Optional[str]
    step_name: str
    artifact_name: str


def parse_step_dot_artifact(expr: str) -> Tuple[str, str]:
    if "." not in expr:
        raise UsageError("Expected <step>.<artifact> (example: train.model)")

    step_name, artifact_name = expr.split(".", 1)
    if not step_name or not step_name.isidentifier():
        raise UsageError("Invalid step name: '%s'" % step_name)
    if not artifact_name:
        raise UsageError("Artifact name is required")

    return step_name, artifact_name


def _parse_flow_run_step(pathspec: str) -> Tuple[str, Optional[str], str]:
    parts = pathspec.split("/")
    if len(parts) != 3:
        raise UsageError(
            "Expected <Flow>/<run|latest>/<step> (example: MyFlow/latest/train)"
        )

    flow_name, run_token, step_name = parts
    if not flow_name or not flow_name.isidentifier():
        raise UsageError("Invalid flow name: '%s'" % flow_name)
    if not step_name or not step_name.isidentifier():
        raise UsageError("Invalid step name: '%s'" % step_name)

    run_id = None if run_token == "latest" else run_token
    if run_id == "":
        raise UsageError("Run id cannot be empty")

    return flow_name, run_id, step_name


def parse_mf_show_args(line: str) -> MFShowSpec:
    try:
        tokens = shlex.split(line)

        if len(tokens) == 2 and "/" in tokens[0]:
            flow_name, run_id, step_name = _parse_flow_run_step(tokens[0])
            artifact_name = tokens[1]
            if not artifact_name:
                raise UsageError("Artifact name is required")
            return MFShowSpec(flow_name, run_id, step_name, artifact_name)

        if len(tokens) == 2 and tokens[0] == "latest":
            step_name, artifact_name = parse_step_dot_artifact(tokens[1])
            return MFShowSpec(None, None, step_name, artifact_name)

        if len(tokens) == 1:
            step_name, artifact_name = parse_step_dot_artifact(tokens[0])
            return MFShowSpec(None, None, step_name, artifact_name)

        raise UsageError(_MF_SHOW_USAGE)
    except ValueError as e:
        raise UsageError("Invalid %mf_show arguments: %s" % e) from e


def _resolve_latest_flow_name() -> str:
    latest = None
    latest_created_at = None

    for flow in Metaflow():
        run = flow.latest_run
        if run is None:
            continue
        if latest_created_at is None or run.created_at > latest_created_at:
            latest = flow
            latest_created_at = run.created_at

    if latest is None:
        raise UsageError(
            "No runs found in the current namespace. Use an explicit flow path "
            "like: %mf_show MyFlow/latest/train model"
        )

    return latest.id


def fetch_artifact(spec: MFShowSpec):
    flow_name = spec.flow_name or _resolve_latest_flow_name()

    try:
        flow = Flow(flow_name)
    except (
        MetaflowInvalidPathspec,
        MetaflowNotFound,
        MetaflowNamespaceMismatch,
    ) as e:
        raise UsageError(
            "Flow '%s' not found or not accessible in the current namespace."
            % flow_name
        ) from e

    if spec.run_id is None:
        run = flow.latest_run
        if run is None:
            raise UsageError("No runs found for flow '%s'." % flow_name)
    else:
        try:
            run = Run("%s/%s" % (flow_name, spec.run_id))
        except (
            MetaflowInvalidPathspec,
            MetaflowNotFound,
            MetaflowNamespaceMismatch,
        ) as e:
            raise UsageError(
                "Run '%s/%s' not found in the current namespace."
                % (flow_name, spec.run_id)
            ) from e

    try:
        step = run[spec.step_name]
    except KeyError as e:
        raise UsageError(
            "Step '%s' not found in run '%s'." % (spec.step_name, run.pathspec)
        ) from e

    tasks = list(step)
    if not tasks:
        raise UsageError(
            "No tasks found for step '%s' in run '%s'." % (spec.step_name, run.pathspec)
        )
    if len(tasks) > 1:
        raise UsageError(
            "Step '%s' in run '%s' has %d tasks. %%mf_show currently supports "
            "single-task steps only." % (spec.step_name, run.pathspec, len(tasks))
        )

    task = tasks[0]
    try:
        value = task[spec.artifact_name].data
    except KeyError as e:
        raise UsageError(
            "Artifact '%s' not found in task '%s'."
            % (spec.artifact_name, task.pathspec)
        ) from e

    return value, run, step, task


def render_artifact(value: Any) -> None:
    try:
        import pandas as pd
    except ImportError:
        pd = None

    if pd is not None and isinstance(value, pd.DataFrame):
        display(value)
        return

    try:
        from matplotlib.figure import Figure
    except ImportError:
        Figure = None

    if Figure is not None and isinstance(value, Figure):
        display(value)
        return

    try:
        import numpy as np
    except ImportError:
        np = None

    if np is not None and isinstance(value, np.ndarray):
        flat = value.ravel()
        preview_size = 20
        preview = flat[:preview_size].tolist()
        payload = {
            "type": "numpy.ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "preview": preview,
            "truncated": bool(flat.size > preview_size),
        }
        display(JSON(payload))
        return

    if isinstance(value, (dict, list)):
        try:
            json.dumps(value)
            display(JSON(value))
        except (TypeError, ValueError):
            print(repr(value))
        return

    print(repr(value))