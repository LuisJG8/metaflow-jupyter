import ast
import hashlib
import linecache
import textwrap

from metaflow.exception import CommandException
from IPython.core.error import UsageError
from IPython.core.magic import Magics, cell_magic, line_magic, magics_class

from .metaflow_jupyter_exec import (
    execute_flow_source,
)
from .metaflow_jupyter_flow import (
    _build_flow_source as _build_flow_source_impl,
    _parse_run_flow_invocation as _parse_run_flow_invocation_impl,
    _validate_run_flow_parameter_overrides as _validate_run_flow_parameter_overrides_impl,
)

from .metaflow_jupyter_step import (
    DECORATOR_MAGIC_NAMES,
    MissingImportError,
    _PreviewSelf,
    _parse_stacked_decorator_cell as _parse_stacked_decorator_cell_impl,
    _parse_step_name as _parse_step_name_impl,
    _register_step_definition as _register_step_definition_impl,
    _validate_no_inline_step_decorators as _validate_no_inline_step_decorators_impl,
)


STEP_REGISTRY = {}  # {"start": {"decorators": ["@retry(times=1)"], "body": "..."}}
PARAM_REGISTRY = {}  # {"alpha": {"source": "...", "required": False, "cell_id": "..."}}
CELL_STEP_OWNERSHIP = {}  # {cell_id: step_name}
CELL_PARAM_OWNERSHIP = {}  # {cell_id: {"alpha", "epochs"}}
CURRENT_CELL_ID = None
ALLOWED_PARAMETER_TYPES = {"str", "int", "float", "bool", "JSONType"}
PREVIEW_SELF = _PreviewSelf()


def _on_pre_run_cell(info):
    global CURRENT_CELL_ID
    CURRENT_CELL_ID = getattr(info, "cell_id", None)


def _on_post_run_cell(result):
    global CURRENT_CELL_ID
    CURRENT_CELL_ID = None


def _parse_step_name(step_name_text):
    return _parse_step_name_impl(step_name_text)


def _validate_no_inline_step_decorators(step_name, step_body):
    return _validate_no_inline_step_decorators_impl(step_name, step_body)


def _parse_stacked_decorator_cell(first_decorator_name, first_decorator_args, cell_text):
    return _parse_stacked_decorator_cell_impl(
        first_decorator_name,
        first_decorator_args,
        cell_text,
    )


def _register_step_definition(shell, step_name, decorators, body):
    return _register_step_definition_impl(
        shell,
        step_name,
        decorators,
        body,
        STEP_REGISTRY,
        CELL_STEP_OWNERSHIP,
        PARAM_REGISTRY,
        CURRENT_CELL_ID,
        PREVIEW_SELF,
    )


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
        raise UsageError(
            "Invalid Parameter type. Supported types: str, int, float, bool, JSONType."
        )

    return candidate


def _extract_parameter_spec(assign_node, source_text):
    if len(assign_node.targets) != 1 or not isinstance(assign_node.targets[0], ast.Name):
        raise UsageError("%%params supports assignments like: name = Parameter(...).")

    param_name = assign_node.targets[0].id
    if not param_name.isidentifier():
        raise UsageError(
            "Parameter name '%s' must be a valid Python identifier." % param_name
        )

    value = assign_node.value
    if not isinstance(value, ast.Call) or not _is_parameter_call(value.func):
        raise UsageError(
            "Invalid definition for parameter '%s'. Use: %s = Parameter(...)."
            % (param_name, param_name)
        )

    if not value.args:
        raise UsageError(
            "Parameter '%s' must include the CLI name as first argument." % param_name
        )

    first_arg = value.args[0]
    if not (isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str)):
        raise UsageError(
            "Parameter '%s' first argument must be a string literal." % param_name
        )

    has_default = False
    has_preview_default = False
    preview_default = None
    uses_json_type = False
    for keyword in value.keywords:
        if keyword.arg is None:
            raise UsageError(
                "Parameter '%s' does not support **kwargs in %%params." % param_name
            )
        if keyword.arg == "default":
            has_default = True
            try:
                preview_default = ast.literal_eval(keyword.value)
                has_preview_default = True
            except (SyntaxError, ValueError):

                has_preview_default = False
                preview_default = None
        elif keyword.arg == "type":
            parsed_type = _parse_parameter_type(keyword.value)
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
    body = textwrap.dedent(cell_text).strip("\n")
    if not body.strip():
        raise UsageError("%%params cell cannot be empty.")

    try:
        parsed = ast.parse(body)
    except SyntaxError as e:
        raise UsageError("Invalid syntax in %%params: %s" % e.msg) from e

    specs = []
    seen = set()
    for stmt in parsed.body:
        if not isinstance(stmt, ast.Assign):
            raise UsageError(
                "%%params only supports Parameter assignments like: alpha = Parameter(...)."
            )
        spec = _extract_parameter_spec(stmt, body)
        if spec["name"] in seen:
            raise UsageError(
                "Duplicate parameter name '%s' in %%params cell." % spec["name"]
            )
        seen.add(spec["name"])
        specs.append(spec)

    return specs


def _register_param_definitions(shell, param_specs):
    incoming_cell_id = CURRENT_CELL_ID
    owned_param_names = set()
    if incoming_cell_id:
        owned_param_names = set(CELL_PARAM_OWNERSHIP.get(incoming_cell_id, set()))

    effective_registry = {
        name: definition
        for name, definition in PARAM_REGISTRY.items()
        if name not in owned_param_names
    }

    for spec in param_specs:
        existing_definition = effective_registry.get(spec["name"])
        if existing_definition is None:
            continue

        existing_cell_id = existing_definition.get("cell_id")
        if incoming_cell_id and existing_cell_id:
            if incoming_cell_id != existing_cell_id:
                raise UsageError(
                    "Duplicate parameter name '%s'. Parameter is already owned by another cell."
                    % spec["name"]
                )
        elif (
            existing_definition.get("source") != spec["source"]
            or bool(existing_definition.get("required")) != spec["required"]
            or bool(existing_definition.get("has_preview_default"))
            != spec["has_preview_default"]
            or bool(existing_definition.get("uses_json_type"))
            != spec["uses_json_type"]
        ):
            raise UsageError(
                "Duplicate parameter name '%s'. A different definition is already "
                "registered for this parameter." % spec["name"]
            )

    if incoming_cell_id:
        for param_name in owned_param_names:
            existing = PARAM_REGISTRY.get(param_name)
            if existing and existing.get("cell_id") == incoming_cell_id:
                PARAM_REGISTRY.pop(param_name, None)
        CELL_PARAM_OWNERSHIP.pop(incoming_cell_id, None)

    new_owned_names = set()
    for spec in param_specs:
        existing_definition = PARAM_REGISTRY.get(spec["name"])
        stored_cell_id = None
        if incoming_cell_id:
            stored_cell_id = incoming_cell_id
        elif existing_definition is not None:
            stored_cell_id = existing_definition.get("cell_id")

        PARAM_REGISTRY[spec["name"]] = {
            "source": spec["source"],
            "required": spec["required"],
            "has_preview_default": spec["has_preview_default"],
            "preview_default": spec["preview_default"],
            "uses_json_type": spec["uses_json_type"],
            "cell_id": stored_cell_id,
        }
        if incoming_cell_id and stored_cell_id == incoming_cell_id:
            new_owned_names.add(spec["name"])

    if incoming_cell_id:
        CELL_PARAM_OWNERSHIP[incoming_cell_id] = new_owned_names

    print("Stored params: %s" % ", ".join(spec["name"] for spec in param_specs))


def _parse_run_flow_invocation(line):
    return _parse_run_flow_invocation_impl(line)


def _validate_run_flow_parameter_overrides(param_overrides):
    return _validate_run_flow_parameter_overrides_impl(param_overrides, PARAM_REGISTRY)


def _build_flow_source(flow_name):
    return _build_flow_source_impl(flow_name, STEP_REGISTRY, PARAM_REGISTRY)


def _build_flow_class(flow_name):
    flow_source = _build_flow_source(flow_name)
    source_hash = hashlib.sha1(flow_source.encode("utf-8")).hexdigest()[:12]
    source_filename = "<metaflow_jupyter:%s:%s>" % (flow_name, source_hash)

    source_lines = flow_source.splitlines(True)
    if not source_lines or not source_lines[-1].endswith("\n"):
        source_lines.append("\n")
    linecache.cache[source_filename] = (
        len(flow_source),
        None,
        source_lines,
        source_filename,
    )

    module_name = "_metaflow_jupyter_%s_%s" % (flow_name.lower(), source_hash)
    namespace = {"__name__": module_name}
    compiled = compile(flow_source, source_filename, "exec")
    exec(compiled, namespace, namespace)
    flow_cls = namespace.get(flow_name)
    if flow_cls is None:
        raise CommandException(
            "Flow source did not define class '%s'." % flow_name
        )
    return flow_cls


def _make_decorator_magic(decorator_name):
    def _decorator_magic(self, line, cell):
        self._handle_decorator_magic(decorator_name, line, cell)

    _decorator_magic.__name__ = decorator_name
    return cell_magic(_decorator_magic)


@magics_class
class MetaflowJupyterMagics(Magics):
    def _handle_decorator_magic(self, decorator_name, line, cell):
        step_name, decorators, body = _parse_stacked_decorator_cell(
            decorator_name,
            line,
            cell,
        )
        _register_step_definition(self.shell, step_name, decorators, body)

    @cell_magic
    def step(self, line, cell):
        name = _parse_step_name(line)
        body = _validate_no_inline_step_decorators(name, cell)
        _register_step_definition(self.shell, name, [], body)

    @cell_magic
    def params(self, line, cell):
        if line.strip():
            raise UsageError("Usage: %%params")
        param_specs = _parse_params_cell(cell)
        _register_param_definitions(self.shell, param_specs)

    for _decorator_name in DECORATOR_MAGIC_NAMES:
        locals()[_decorator_name] = _make_decorator_magic(_decorator_name)
    del _decorator_name

    @line_magic
    def run_flow(self, line):
        try:
            flow_name, param_overrides = _parse_run_flow_invocation(line)
            _validate_run_flow_parameter_overrides(param_overrides)
            flow_source = _build_flow_source(flow_name)
        except CommandException as e:
            raise UsageError(str(e)) from e

        return execute_flow_source(flow_name, flow_source, param_overrides)


def load_ipython_extension(ipython):
    ipython.register_magics(MetaflowJupyterMagics)
    events = getattr(ipython, "events", None)
    if events is not None:
        events.register("pre_run_cell", _on_pre_run_cell)
        events.register("post_run_cell", _on_post_run_cell)