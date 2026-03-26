import ast
import shlex

from metaflow.exception import CommandException

from .metaflow_jupyter_step import (
    _extract_next_step_targets,
    _extract_step_decorator_name,
    _indent_with_spaces,
    validate_step_body_before_exec,
)


def _parse_run_flow_invocation(line):
    try:
        tokens = shlex.split(line)
    except ValueError as e:
        raise CommandException("Invalid %run_flow arguments: %s" % e) from e

    if not tokens:
        raise CommandException("Usage: %run_flow <FlowName>")

    flow_name = tokens[0]
    overrides = {}
    for token in tokens[1:]:
        if "=" not in token:
            raise CommandException(
                "Invalid parameter override '%s'. Use key=value." % token
            )
        key, raw_value = token.split("=", 1)
        if not key or not key.isidentifier():
            raise CommandException(
                "Invalid parameter name '%s' in %run_flow overrides." % key
            )
        if key in overrides:
            raise CommandException(
                "Duplicate parameter override '%s' in %run_flow." % key
            )
        try:
            overrides[key] = ast.literal_eval(raw_value)
        except (SyntaxError, ValueError):
            overrides[key] = raw_value

    return flow_name, overrides


def _validate_run_flow_parameter_overrides(param_overrides, param_registry):
    unknown = sorted(
        param_name for param_name in param_overrides if param_name not in param_registry
    )
    if unknown:
        raise CommandException("Unknown parameter override(s): %s" % ", ".join(unknown))

    missing_required = sorted(
        param_name
        for param_name, definition in param_registry.items()
        if definition.get("required") and param_name not in param_overrides
    )
    if missing_required:
        raise CommandException(
            "Missing required parameter override(s) in %%run_flow: %s"
            % ", ".join(missing_required)
        )


def _build_flow_source(flow_name, step_registry, param_registry):
    if not flow_name or not flow_name.isidentifier():
        raise CommandException("Usage: %run_flow <FlowName>")

    if not step_registry:
        raise CommandException("No steps found. Add steps using %%step <step_name>.")

    if "start" not in step_registry:
        raise CommandException("Missing required step 'start'.")
    if "end" not in step_registry:
        raise CommandException("Missing required step 'end'.")

    step_specs = []
    used_step_decorators = set()

    for step_name, step_data in step_registry.items():
        if not step_name.isidentifier():
            raise CommandException("Invalid step name: %s" % step_name)

        step_decorators = []
        step_body = step_data
        if isinstance(step_data, dict):
            step_decorators = step_data.get("decorators") or []
            step_body = step_data.get("body", "")

        for decorator_line in step_decorators:
            decorator_name = _extract_step_decorator_name(step_name, decorator_line)
            used_step_decorators.add(decorator_name)

        body = validate_step_body_before_exec(step_name, step_body)
        if not body.strip():
            body = "pass"

        next_targets = _extract_next_step_targets(step_name, body)
        step_specs.append((step_name, step_decorators, body, next_targets))

    step_index = {name: idx for idx, (name, _, _, _) in enumerate(step_specs)}
    for step_name, _, _, next_targets in step_specs:
        for target_name in next_targets:
            if target_name not in step_index:
                raise CommandException(
                    "Step '%s' references unknown step '%s' in self.next(...)."
                    % (step_name, target_name)
                )
            if step_index[target_name] <= step_index[step_name]:
                raise CommandException(
                    "Step '%s' references previous step '%s' in self.next(...). "
                    "Steps must transition to steps defined later in the flow."
                    % (step_name, target_name)
                )

    used_parameter_imports = set()
    if param_registry:
        used_parameter_imports.add("Parameter")
        if any(spec.get("uses_json_type") for spec in param_registry.values()):
            used_parameter_imports.add("JSONType")

    import_items = ["FlowSpec", "step"] + sorted(
        used_step_decorators | used_parameter_imports
    )
    parts = [
        "from metaflow import %s" % ", ".join(import_items),
        "",
        "class %s(FlowSpec):" % flow_name,
    ]

    for param_spec in param_registry.values():
        parts.extend(["", _indent_with_spaces(param_spec["source"], 4)])

    for step_name, step_decorators, body, _ in step_specs:
        parts.extend([""])
        parts.extend("    %s" % deco for deco in step_decorators)
        parts.extend(
            [
                "    @step",
                "    def %s(self):" % step_name,
                _indent_with_spaces(body, 8),
            ]
        )

    parts.extend(["", "if __name__ == '__main__':", "    %s()" % flow_name])
    return ("\n".join(parts)).rstrip() + "\n"