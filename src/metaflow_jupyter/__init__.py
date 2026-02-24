"""Jupyter-native tools for Metaflow."""

__version__ = "0.1.0"


def load_ipython_extension(ipython):
    """Called by %load_ext metaflow_jupyter."""
    from .magics import register_magics

    register_magics(ipython)
