import ast
import os

from metaflow._vendor import click

from metaflow.runner.metaflow_jupyter_export import (
    NotebookExportError,
    build_flow_source_from_notebook,
    derive_output_filename_from_notebook,
)

EXPORT_OUTPUT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "runner",
        "jupyter-notebook",
    )
)


@click.command(help="Export a notebook with %%step cells to a standalone flow file.")
@click.argument("notebook", type=click.Path(exists=True, dir_okay=False, resolve_path=True),)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, resolve_path=True),
    default=None,
    help=("Output filename for the generated flow .py file." "Saved to metaflow/runner/jupyter-notebook."
    ),
)
@click.option(
    "--flow-name",
    default=None,
    help="Optional FlowSpec class name. Defaults to one derived from the notebook filename.",
)
def main(notebook, output, flow_name):
    try:
        result = build_flow_source_from_notebook(notebook, flow_name=flow_name)
        ast.parse(result.source)
        output_filename = derive_output_filename_from_notebook(notebook)
    except (NotebookExportError, SyntaxError) as e:
        raise click.ClickException(str(e))

    if output is None:
        notebook_dir = os.path.dirname(os.path.abspath(notebook))
        output = os.path.join(notebook_dir, output_filename)
    else:
        os.makedirs(EXPORT_OUTPUT_DIR, exist_ok=True)
        output = os.path.join(EXPORT_OUTPUT_DIR, output_filename)

    for warning in result.warnings:
        click.secho("Warning: %s" % warning, fg="yellow", err=True)

    try:
        with open(output, "w", encoding="utf-8") as f:
            f.write(result.source)
    except OSError as e:
        raise click.ClickException("Failed to write output file '%s': %s" % (output, e)) from e

    click.secho(
        "Exported flow '%s' to %s (%d steps, %d params)."
        % (result.flow_name, output, result.step_count, result.param_count),
        fg="green",
    )


if __name__ == "__main__":
    main()
