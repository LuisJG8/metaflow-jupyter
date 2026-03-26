import os
import tempfile
import threading
from queue import Queue, Empty

from metaflow import Runner


def execute_flow_source(flow_name, flow_source, param_overrides, base_dir=None):
    if base_dir is None:
        base_dir = os.getcwd()

    runner = None
    flow_file_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=flow_name,
            suffix=".py",
            mode="w",
            dir=base_dir,
            delete=False,
        ) as tmp_flow_file:
            tmp_flow_file.write(flow_source)
            tmp_flow_file.flush()
            flow_file_path = tmp_flow_file.name

        env = os.environ.copy()
        env["JPY_PARENT_PID"] = ""

        runner = Runner(
            flow_file=flow_file_path,
            show_output=True,
            env=env,
            cwd=base_dir,
        )
        result = runner.run(**param_overrides)
        return result.run
    finally:
        if runner is not None:
            runner.cleanup()
        if flow_file_path and os.path.exists(flow_file_path):
            os.unlink(flow_file_path)