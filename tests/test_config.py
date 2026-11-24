"""Path resolution must be relative to the project and overrideable."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from table_vlm import paths
from table_vlm.demo import DemoService


class PathTest(unittest.TestCase):
    def test_defaults_are_inside_the_project(self):
        self.assertEqual(paths.model_path(), (paths.PROJECT_ROOT / 'models/Qwen3-VL-2B-Instruct').resolve())
        self.assertEqual(paths.adapter_path(), (paths.PROJECT_ROOT / 'artifacts/q1-adapter').resolve())
        self.assertEqual(paths.output_path(), (paths.PROJECT_ROOT / 'demo-output').resolve())
        self.assertEqual(paths.gpu_python(), Path(sys.executable).resolve())

    def test_environment_and_explicit_override(self):
        with tempfile.TemporaryDirectory() as temp:
            env_value = Path(temp) / 'from-env'
            explicit = Path(temp) / 'explicit'
            with mock.patch.dict(os.environ, {paths.ENV_MODEL: str(env_value)}):
                self.assertEqual(paths.model_path(), env_value.resolve())
                self.assertEqual(paths.model_path(explicit), explicit.resolve())
            with mock.patch.dict(os.environ, {}, clear=True):
                self.assertEqual(paths.model_path(), (paths.PROJECT_ROOT / 'models/Qwen3-VL-2B-Instruct').resolve())

    def test_extra_pythonpath_accepts_multiple_directories(self):
        with tempfile.TemporaryDirectory() as temp:
            first, second = Path(temp) / 'a', Path(temp) / 'b'
            value = os.pathsep.join([str(first), str(second)])
            with mock.patch.dict(os.environ, {paths.ENV_EXTRA_PYTHONPATH: value}):
                self.assertEqual(paths.extra_pythonpath(), [first.resolve(), second.resolve()])
            self.assertEqual(paths.extra_pythonpath(str(first)), [first.resolve()])

    def test_demo_service_uses_portable_defaults_and_command(self):
        service = DemoService(device='cpu')
        self.assertEqual(service.model, paths.model_path())
        self.assertEqual(service.adapter, paths.adapter_path())
        self.assertEqual(service.python, Path(sys.executable).resolve())
        command = service.inference_command(Path('in.jsonl'), Path('out.jsonl'))
        self.assertIn('table_vlm.demo_infer', command)
        self.assertEqual(command[command.index('--device') + 1], 'cpu')
        self.assertEqual(command[command.index('--min-pixels') + 1], '65536')
        self.assertEqual(command[command.index('--max-pixels') + 1], '196608')


if __name__ == '__main__':
    unittest.main()
