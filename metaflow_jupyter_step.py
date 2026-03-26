import ast
import copy
import textwrap

try:
    from IPython.core.error import UsageError
except ImportError:
    class UsageError(Exception):
        pass


DECORATOR_MAGIC_NAMES = ("environment","batch","card","catch","conda","pypi",
                         "kubernetes","resources","retry","secrets","timeout")
ALLOWED_STEP_DECORATORS = set(DECORATOR_MAGIC_NAMES)


class MissingImportError(UsageError):
    pass

# provides a fake self so code like self.x = ... can run safely.
class _PreviewSelf(object):
    def next(self, *args, **kwargs):
        return None


# In preview mode, execute the cell quickly without running the whole flow.
# Removes self.next(...) so no graph transition happens.
class _StripNextCalls(ast.NodeTransformer):
    def visit_Call(self, node):
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
            and node.func.attr == "next"
        ):
            return ast.Constant(value=None)
        return self.generic_visit(node)


def _indent_with_spaces(block, spaces):
    prefix = " " * spaces
    return "\n".join((prefix + line if line else prefix) for line in block.splitlines())


def _parse_step_name(step_name_text):
    step_name = step_name_text.strip()
    if not step_name:
        raise UsageError("Usage: %%step <step_name>")
    if not step_name.isidentifier():
        raise UsageError("Step name must be a valid Python identifier")
    return step_name


def _parse_step_body(step_name, step_body):
    body = textwrap.dedent(step_body).strip("\n")
    try:
        parsed_body = ast.parse(body)
    except SyntaxError as e:
        raise UsageError("Invalid syntax in step '%s': %s" % (step_name, e.msg)) from e
    return body, parsed_body


def validate_self_next(node):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "next"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
    )

def validate_self_next_argument(node):
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and node.attr.isidentifier()
    )

def validate_step_body_before_exec(step_name, step_body):
    body, parsed_body = _parse_step_body(step_name, step_body)

    if step_name != "end":
        has_top_level_next_call = any(
            isinstance(stmt, ast.Expr) and validate_self_next(stmt.value)
            for stmt in parsed_body.body
        )
        if not has_top_level_next_call:
            raise UsageError("Step '%s' is missing self.next(...)." % step_name)

        last_stmt = parsed_body.body[-1] if parsed_body.body else None
        if not (isinstance(last_stmt, ast.Expr) and validate_self_next(last_stmt.value)):
            raise UsageError(
                "Step '%s' must end with self.next(...). No code can appear "
                "after self.next(...)." % step_name
            )

        next_call = last_stmt.value
        if not next_call.args or not all(
            validate_self_next_argument(arg) for arg in next_call.args
        ):
            raise UsageError(
                "Step '%s' has invalid self.next(...). Use self.next(self.<next_step>) "
                "with one or more step references." % step_name
            )

    return body


def _extract_next_step_targets(step_name, step_body):
    if step_name == "end":
        return []

    _, parsed_body = _parse_step_body(step_name, step_body)
    if not parsed_body.body:
        return []

    last_stmt = parsed_body.body[-1]
    if not (isinstance(last_stmt, ast.Expr) and validate_self_next(last_stmt.value)):
        return []

    return [
        arg.attr for arg in last_stmt.value.args if validate_self_next_argument(arg)
    ]


def _validate_no_inline_step_decorators(step_name, step_body):
    body = textwrap.dedent(step_body).strip("\n")
    if not body:
        return body

    for line in body.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("@"):
            raise UsageError(
                "Inline decorators are not supported in %%%%step '%s'. "
                "Use %%%%retry / %%%%batch style decorator magics before %%%%step." % step_name
            )
        break
    return body


def _parse_decorator_magic_arguments(decorator_name, raw_args):
    args_text = raw_args.strip()
    if not args_text:
        return ""

    try:
        call_node = ast.parse("f(%s)" % args_text, mode="eval").body
    except SyntaxError as e:
        raise UsageError("Invalid arguments for %%%s: %s" % (decorator_name, e.msg)) from e

    if not isinstance(call_node, ast.Call):
        raise UsageError(
            "Invalid arguments for %%%s. Use keyword arguments like: "
            "%%%s key=value" % (decorator_name, decorator_name)
        )
    if call_node.args:
        raise UsageError(
            "Decorator magic %%%s supports keyword arguments only." % decorator_name
        )
    if any(keyword.arg is None for keyword in call_node.keywords):
        raise UsageError("Decorator magic %%%s does not support **kwargs." % decorator_name)

    return ", ".join(
        "%s=%s" % (keyword.arg, ast.unparse(keyword.value)) for keyword in call_node.keywords
    )


def _decorator_magic_to_source(decorator_name, raw_args):
    if decorator_name not in ALLOWED_STEP_DECORATORS:
        raise UsageError("Unknown decorator magic '%%%%%s'." % decorator_name)

    normalized_args = _parse_decorator_magic_arguments(decorator_name, raw_args)
    if not normalized_args:
        return "@%s" % decorator_name
    return "@%s(%s)" % (decorator_name, normalized_args)


def _extract_step_decorator_name(step_name, decorator_line):
    if not decorator_line.startswith("@"):
        raise UsageError(
            "Invalid decorator syntax in step '%s': '%s'" % (step_name, decorator_line)
        )

    decorator_expr = decorator_line[1:].strip()
    if not decorator_expr:
        raise UsageError(
            "Invalid decorator syntax in step '%s': '%s'" % (step_name, decorator_line)
        )

    try:
        parsed_expr = ast.parse(decorator_expr, mode="eval")
    except SyntaxError as e:
        raise UsageError(
            "Invalid decorator syntax in step '%s': %s" % (step_name, e.msg)
        ) from e

    deco_node = parsed_expr.body
    if isinstance(deco_node, ast.Call):
        deco_node = deco_node.func

    if isinstance(deco_node, ast.Name):
        decorator_name = deco_node.id
    elif isinstance(deco_node, ast.Attribute):
        decorator_name = deco_node.attr
    else:
        raise UsageError(
            "Invalid decorator syntax in step '%s': '%s'" % (step_name, decorator_line)
        )

    if decorator_name == "step":
        raise UsageError(
            "Do not use @step inside %%step '%s'. @step is added automatically." % step_name
        )

    if decorator_name not in ALLOWED_STEP_DECORATORS:
        raise UsageError(
            "Unknown step decorator '%s' in step '%s'." % (decorator_name, step_name)
        )

    return decorator_name


def _parse_stacked_decorator_cell(first_decorator_name, first_decorator_args, cell_text):
    decorator_lines = [
        _decorator_magic_to_source(first_decorator_name, first_decorator_args)
    ]
    lines = textwrap.dedent(cell_text).splitlines()
    idx = 0

    while idx < len(lines):
        stripped = lines[idx].strip()
        if not stripped:
            idx += 1
            continue

        if not stripped.startswith("%%"):
            raise UsageError("Decorator magics must be followed by %%step <step_name>.")

        magic_line = stripped[2:].strip()
        if not magic_line:
            raise UsageError("Invalid magic syntax: '%s'" % stripped)

        parts = magic_line.split(None, 1)
        magic_name = parts[0]
        magic_args = parts[1] if len(parts) > 1 else ""

        if magic_name == "step":
            step_name = _parse_step_name(magic_args)
            body = "\n".join(lines[idx + 1 :]).strip("\n")
            return step_name, decorator_lines, body

        if magic_name not in ALLOWED_STEP_DECORATORS:
            raise UsageError("Unknown decorator magic '%%%%%s'." % magic_name)

        decorator_lines.append(_decorator_magic_to_source(magic_name, magic_args))
        idx += 1

    raise UsageError("Decorator magics must include %%step <step_name>.")


def _execute_step_preview(shell, step_name, step_body, param_registry, preview_self):
    body, parsed_body = _parse_step_body(step_name, step_body)
    if not body.strip():
        return

    preview_tree = _StripNextCalls().visit(parsed_body)
    ast.fix_missing_locations(preview_tree)
    preview_code = compile(preview_tree, "<metaflow-step-preview>", "exec")

    for param_name, param_spec in param_registry.items():
        if param_spec.get("has_preview_default"):
            try:
                value = copy.deepcopy(param_spec.get("preview_default"))
            except Exception:
                value = param_spec.get("preview_default")
        else:
            value = None
        setattr(preview_self, param_name, value)

    namespace = {"__builtins__": __builtins__, "self": preview_self}
    try:
        exec(preview_code, namespace, namespace)
    except NameError as e:
        # Step previews run in an isolated namespace so every dependency must
        # be imported/defined in the same cell. Missing names are treated as
        # missing imports to make notebook behavior explicit and deterministic.
        message = str(e)
        if message.startswith("name '") and "' is not defined" in message:
            missing_name = message.split("name '", 1)[1].split("' is not defined", 1)[0]
            raise MissingImportError(
                "Missing import in step '%s': '%s' is not defined. "
                "Add the import/definition in this %%step cell."
                % (step_name, missing_name)
            ) from e
        raise


def _register_step_definition(
    shell, step_name, decorators, body, step_registry, 
    cell_step_ownership, param_registry, current_cell_id,preview_self,
):
    validated_body = validate_step_body_before_exec(step_name, body)
    incoming_cell_id = current_cell_id
    if incoming_cell_id:
        owned_step_name = cell_step_ownership.get(incoming_cell_id)
        if owned_step_name and owned_step_name != step_name:
            owned_step = step_registry.get(owned_step_name)
            if owned_step and owned_step.get("cell_id") == incoming_cell_id:
                step_registry.pop(owned_step_name, None)
            cell_step_ownership.pop(incoming_cell_id, None)

    existing_definition = step_registry.get(step_name)
    if existing_definition is not None:
        existing_cell_id = existing_definition.get("cell_id")
        existing_decorators = existing_definition.get("decorators") or []
        existing_body = textwrap.dedent(existing_definition.get("body", "")).strip("\n")

        if incoming_cell_id and existing_cell_id:
            if incoming_cell_id != existing_cell_id:
                raise UsageError(
                    "Duplicate step name '%s'. Step is already owned by another cell." % step_name
                )
        elif existing_decorators != decorators or existing_body != validated_body:
            raise UsageError(
                "Duplicate step name '%s'. A different definition is already "
                "registered for this step name." % step_name
            )

    _execute_step_preview(shell, step_name, validated_body, param_registry, preview_self)
    is_update = existing_definition is not None
    stored_cell_id = None
    if incoming_cell_id:
        stored_cell_id = incoming_cell_id
    elif existing_definition is not None:
        stored_cell_id = existing_definition.get("cell_id")

    step_registry[step_name] = {
        "decorators": decorators,
        "body": validated_body,
        "cell_id": stored_cell_id,
    }
    if stored_cell_id:
        cell_step_ownership[stored_cell_id] = step_name

    if is_update:
        print("Updated step: %s" % step_name)
    else:
        print("Stored step: %s" % step_name)