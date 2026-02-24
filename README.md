# metaflow-jupyter

Jupyter-native tools for [Metaflow](https://metaflow.org). Define flows across notebook cells, visualize DAGs inline, inspect artifacts interactively, and export notebooks to production flow files.

> **Status:** Early development. See [issues](https://github.com/npow/metaflow-jupyter/issues) for planned work.

## Planned Features

- **`%%step` magic** — define Metaflow steps across multiple notebook cells instead of cramming everything into one
- **`%mf_show` magic** — render artifacts (DataFrames, plots, models) inline with type-aware display
- **DAG widget** — interactive flow graph visualization with live step status
- **`mf-export` CLI** — convert a notebook with `%%step` cells into a standalone `.py` flow file
- **Better errors** — clear, actionable error messages when things go wrong in notebooks

## Install

```bash
pip install metaflow-jupyter  # not yet published
```

```python
%load_ext metaflow_jupyter
```

## Quick Example

```python
# Cell 1
%%step start
self.data = [1, 2, 3]
self.next(self.end)

# Cell 2
%%step end
print(f"Got {len(self.data)} items")

# Cell 3
%run_flow MyFlow
```

## Requirements

- Python 3.9+
- Metaflow
- IPython / Jupyter

## Contributing

This project is part of [GSoC 2026](https://github.com/npow/metaflow-jupyter/issues/1). Check the [issues](https://github.com/npow/metaflow-jupyter/issues) for starter tasks.

## License

Apache 2.0
