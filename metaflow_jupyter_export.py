import ast
import keyword
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import nbformat

from metaflow.exception import CommandException

from .metaflow_jupyter_flow import _build_flow_source
from .metaflow_jupyter_sql import parse_sql_step_definition
from .metaflow_jupyter_step import (
    DECORATOR_MAGIC_NAMES,
    _parse_stacked_step_or_sql_cell,
    _parse_step_name,
    _validate_no_inline_step_decorators,
    validate_step_body_before_exec,
)


ALLOWED_PARAMETER_TYPES = {"str", "int", "float", "bool", "JSONType"}
HELPER_NODE_TYPES = (
    ast.Import,
    ast.ImportFrom,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Assign,
    ast.AnnAssign,
)


@dataclass(frozen=True)
class ExportResult:
    flow_name: str
    source: str
    warnings: List[str]
    step_count: int
    param_count: int


class NotebookExportError(Exception):
    pass


def _validate_flow_name(flow_name: str) -> str:
    if not flow_name:
        raise NotebookExportError("Flow name cannot be empty.")
    if not flow_name.isidentifier() or keyword.iskeyword(flow_name):
        raise NotebookExportError(
            "Invalid flow name '%s'. Use --flow-name with a valid Python identifier." % flow_name
        )
    return flow_name


def _derive_flow_name(notebook_path: str) -> str:
    stem = os.path.splitext(os.path.basename(notebook_path))[0]
    tokens = [token for token in re.split(r"[^0-9A-Za-z]+", stem) if token]

    if tokens:
        flow_name = "".join(token[:1].upper() + token[1:] for token in tokens)
    else:
        flow_name = "Notebook"

    if not flow_name.endswith("Flow"):
        flow_name += "Flow"
    if flow_name[0].isdigit():
        flow_name = "Notebook" + flow_name

    return _validate_flow_name(flow_name)


def _normalized_flow_name(notebook_path: str, flow_name: Optional[str]) -> str:
    if flow_name is not None:
        return _validate_flow_name(flow_name)
    return _derive_flow_name(notebook_path)


def derive_output_filename_from_notebook(notebook_path: str) -> str:
    stem = os.path.splitext(os.path.basename(notebook_path))[0]
    if not stem:
        raise NotebookExportError(
            "Could not derive output filename from notebook path '%s'." % notebook_path
        )
    return "%s.py" % stem


def _parse_magic_line(stripped_line: str):
    magic_line = stripped_line[2:].strip()
    if not magic_line:
        raise NotebookExportError("Invalid magic syntax: '%s'" % stripped_line)
    parts = magic_line.split(None, 1)
    magic_name = parts[0]
    magic_args = parts[1] if len(parts) > 1 else ""
    return magic_name, magic_args


def _first_nonempty_line(lines):
    for idx, line in enumerate(lines):
        if line.strip():
            return idx, line
    return None, None


def _strip_line_magics(cell_source: str, cell_index: int, warnings: List[str]) -> str:
    kept_lines = []
    for line in cell_source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("%") and not stripped.startswith("%%"):
            warnings.append(
                "Skipped line magic in code cell %d: %s" % (cell_index, stripped)
            )
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines).strip("\n")


def _extract_helper_blocks(cell_source: str, cell_index: int) -> List[str]:
    try:
        parsed = ast.parse(cell_source)
    except SyntaxError as e:
        raise NotebookExportError(
            "Invalid syntax in non-step code cell %d: %s" % (cell_index, e.msg)
        ) from e

    helper_blocks = []
    for stmt in parsed.body:
        if not isinstance(stmt, HELPER_NODE_TYPES):
            continue
        source = ast.get_source_segment(cell_source, stmt)
        if source is None:
            source = ast.unparse(stmt)
        helper_blocks.append(source.rstrip())
    return helper_blocks


def _register_step(
    step_registry: Dict[str, Dict[str, object]],
    step_name: str,
    decorators: List[str],
    body: str,
    extra_step_fields: Optional[Dict[str, object]] = None,
):
    validated_body = validate_step_body_before_exec(step_name, body)
    existing = step_registry.get(step_name)
    normalized_decorators = list(decorators or [])
    normalized_extra_step_fields = dict(extra_step_fields or {})

    if existing is not None:
        existing_decorators = list(existing.get("decorators") or [])
        existing_body = existing.get("body", "")
        existing_extra_step_fields = {
            key: value
            for key, value in existing.items()
            if key not in ("decorators", "body")
        }
        if (
            existing_decorators != normalized_decorators
            or existing_body != validated_body
            or existing_extra_step_fields != normalized_extra_step_fields
        ):
            raise NotebookExportError(
                "Duplicate step name '%s'. A different definition is already "
                "registered for this step name." % step_name
            )
        return

    step_definition = {
        "decorators": normalized_decorators,
        "body": validated_body,
    }
    step_definition.update(normalized_extra_step_fields)
    step_registry[step_name] = step_definition


def _is_parameter_call(func_node):
    return (
        isinstance(func_node, ast.Name)
        and func_node.id == "Parameter"
    ) or (
        isinstance(func_node, ast.Attribute)
        and isinstance(func_node.value, ast.Name)
        and func_node.value.id == "metaflow"
        and func_node.attr == "Parameter"
    )


def _parse_parameter_type(type_node):
    if isinstance(type_node, ast.Name):
        candidate = type_node.id
    elif (
        isinstance(type_node, ast.Attribute)
        and isinstance(type_node.value, ast.Name)
        and type_node.value.id == "metaflow"
    ):
        candidate = type_node.attr
    else:
        candidate = None

    if candidate not in ALLOWED_PARAMETER_TYPES:
        raise NotebookExportError(
            "Invalid Parameter type. Supported types: str, int, float, bool, JSONType."
        )
    return candidate


def _extract_parameter_spec(assign_node, source_text):
    if len(assign_node.targets) != 1 or not isinstance(assign_node.targets[0], ast.Name):
        raise NotebookExportError(
            "%%params supports assignments like: name = Parameter(...)."
        )

    param_name = assign_node.targets[0].id
    if not param_name.isidentifier():
        raise NotebookExportError(
            "Parameter name '%s' must be a valid Python identifier." % param_name
        )

    value = assign_node.value
    if not isinstance(value, ast.Call) or not _is_parameter_call(value.func):
        raise NotebookExportError(
            "Invalid definition for parameter '%s'. Use: %s = Parameter(...)."
            % (param_name, param_name)
        )

    if not value.args:
        raise NotebookExportError(
            "Parameter '%s' must include the CLI name as first argument." % param_name
        )

    first_arg = value.args[0]
    if not (isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str)):
        raise NotebookExportError(
            "Parameter '%s' first argument must be a string literal." % param_name
        )

    has_default = False
    has_preview_default = False
    preview_default = None
    uses_json_type = False
    for keyword_node in value.keywords:
        if keyword_node.arg is None:
            raise NotebookExportError(
                "Parameter '%s' does not support **kwargs in %%params." % param_name
            )
        if keyword_node.arg == "default":
            has_default = True
            try:
                preview_default = ast.literal_eval(keyword_node.value)
                has_preview_default = True
            except (SyntaxError, ValueError):
                has_preview_default = False
                preview_default = None
        elif keyword_node.arg == "type":
            parsed_type = _parse_parameter_type(keyword_node.value)
            if parsed_type == "JSONType":
                uses_json_type = True

    source = ast.get_source_segment(source_text, assign_node)
    if source is None:
        source = ast.unparse(assign_node)

    return {
        "name": param_name,
        "source": source.strip(),
        "required": not has_default,
        "has_preview_default": has_preview_default,
        "preview_default": preview_default,
        "uses_json_type": uses_json_type,
    }


def _parse_params_cell(cell_text):
    body = cell_text.strip("\n")
    if not body.strip():
        raise NotebookExportError("%%params cell cannot be empty.")
    try:
        parsed = ast.parse(body)
    except SyntaxError as e:
        raise NotebookExportError("Invalid syntax in %%params: %s" % e.msg) from e

    specs = []
    seen = set()
    for stmt in parsed.body:
        if not isinstance(stmt, ast.Assign):
            raise NotebookExportError(
                "%%params only supports Parameter assignments like: alpha = Parameter(...)."
            )
        spec = _extract_parameter_spec(stmt, body)
        if spec["name"] in seen:
            raise NotebookExportError(
                "Duplicate parameter name '%s' in %%params cell." % spec["name"]
            )
        seen.add(spec["name"])
        specs.append(spec)
    return specs


def _register_param_definitions(param_registry, param_specs):
    for spec in param_specs:
        existing = param_registry.get(spec["name"])
        if existing is not None:
            if (
                existing.get("source") != spec["source"]
                or bool(existing.get("required")) != spec["required"]
                or bool(existing.get("has_preview_default"))
                != spec["has_preview_default"]
                or bool(existing.get("uses_json_type")) != spec["uses_json_type"]
            ):
                raise NotebookExportError(
                    "Duplicate parameter name '%s'. A different definition is already "
                    "registered for this parameter." % spec["name"]
                )
            continue

        param_registry[spec["name"]] = {
            "source": spec["source"],
            "required": spec["required"],
            "has_preview_default": spec["has_preview_default"],
            "preview_default": spec["preview_default"],
            "uses_json_type": spec["uses_json_type"],
        }


def _parse_sql_step_definition_for_export(line, cell_text):
    return parse_sql_step_definition(line, cell_text, NotebookExportError)


def build_flow_source_from_notebook(
    notebook_path: str, flow_name: Optional[str] = None) -> ExportResult:
    effective_flow_name = _normalized_flow_name(notebook_path, flow_name)
    try:
        notebook = nbformat.read(notebook_path, as_version=4)
    except Exception as e:
        raise NotebookExportError(
            "Failed to read notebook '%s': %s" % (notebook_path, e)
        ) from e

    warnings: List[str] = []
    preamble_blocks: List[str] = []
    step_registry: Dict[str, Dict[str, object]] = {}
    param_registry: Dict[str, Dict[str, object]] = {}

    for cell_index, cell in enumerate(notebook.cells, start=1):
        if cell.get("cell_type") != "code":
            continue

        cell_source = (cell.get("source") or "").strip("\n")
        if not cell_source.strip():
            continue

        lines = cell_source.splitlines()
        first_idx, first_line = _first_nonempty_line(lines)
        if first_line is None:
            continue

        stripped_first_line = first_line.strip()

        if stripped_first_line.startswith("%%"):
            try:
                magic_name, magic_args = _parse_magic_line(stripped_first_line)
                remaining_body = "\n".join(lines[first_idx + 1 :])

                if magic_name == "step":
                    step_name = _parse_step_name(magic_args)
                    body = _validate_no_inline_step_decorators(step_name, remaining_body)
                    _register_step(step_registry, step_name, [], body)
                elif magic_name == "sql_step":
                    step_name, compiled_body, metadata = _parse_sql_step_definition_for_export(
                        magic_args,
                        remaining_body,
                    )
                    _register_step(
                        step_registry,
                        step_name,
                        [],
                        compiled_body,
                        extra_step_fields=metadata,
                    )
                elif magic_name == "params":
                    if magic_args.strip():
                        raise NotebookExportError("Usage: %%params")
                    param_specs = _parse_params_cell(remaining_body)
                    _register_param_definitions(param_registry, param_specs)
                elif magic_name in DECORATOR_MAGIC_NAMES:
                    (
                        step_name,
                        decorators,
                        body,
                        extra_step_fields,
                        _skip_preview,
                    ) = _parse_stacked_step_or_sql_cell(
                        magic_name,
                        magic_args,
                        remaining_body,
                        _parse_sql_step_definition_for_export,
                    )
                    _register_step(
                        step_registry,
                        step_name,
                        decorators,
                        body,
                        extra_step_fields=extra_step_fields,
                    )
                else:
                    warnings.append(
                        f"Skipped unsupported cell magic in code cell {cell_index}: %%{magic_name}"
                    )
            except NotebookExportError:
                raise
            except Exception as e:
                raise NotebookExportError(
                    "Error in code cell %d: %s" % (cell_index, e)
                ) from e
            continue

        stripped_cell = _strip_line_magics(cell_source, cell_index, warnings)
        if not stripped_cell.strip():
            continue
        preamble_blocks.extend(_extract_helper_blocks(stripped_cell, cell_index))

    try:
        try:
            source = _build_flow_source(
                effective_flow_name,
                step_registry,
                param_registry,
                preamble_blocks=preamble_blocks,
            )
        except TypeError as e:
            # Backward-compatible fallback for runtimes where _build_flow_source
            # does not yet accept preamble blocks.
            if "preamble_blocks" not in str(e):
                raise
            source = _build_flow_source(
                effective_flow_name,
                step_registry,
                param_registry,
            )
    except CommandException as e:
        raise NotebookExportError(str(e)) from e

    return ExportResult(
        flow_name=effective_flow_name,
        source=source,
        warnings=warnings,
        step_count=len(step_registry),
        param_count=len(param_registry),
    )
