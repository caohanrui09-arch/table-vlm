"""The demo worker boundary is testable without loading a 4.25GB model."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from table_vlm import paths
from table_vlm.demo import DemoService


def make_environment(root: Path) -> tuple[Path, Path, str]:
    model = root / 'model'
    model.mkdir()
    for name in ('config.json', 'preprocessor_config.json', 'tokenizer_config.json', 'tokenizer.json'):
        (model / name).write_text('{}', encoding='utf-8')
    (model / 'model.safetensors').write_bytes(b'fake-weights')
    (model / 'source_receipt.json').write_text(json.dumps({'revision': 'test-revision'}),
                                               encoding='utf-8')
    adapter = root / 'adapter'
    adapter.mkdir()
    (adapter / 'adapter_model.safetensors').write_bytes(b'fake-adapter')
    (adapter / 'adapter_config.json').write_text('{}', encoding='utf-8')
    digest = hashlib.sha256(b'fake-adapter').hexdigest()
    return model, adapter, digest


class DemoServiceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.model, self.adapter, self.digest = make_environment(self.root)
        self.output = self.root / 'output'
        self.image = self.root / 'table.png'
        Image.new('RGB', (32, 24), 'white').save(self.image)
        self.service = DemoService(output=self.output, model=self.model, adapter=self.adapter,
                                   python=Path(sys.executable), device='cpu',
                                   adapter_sha=self.digest, base_revision='test-revision')

    def tearDown(self):
        self.temp.cleanup()

    def stub(self, html: str, status: str = 'ok', seen: dict | None = None):
        def fake(command, folder, env):
            if seen is not None:
                seen['env'] = env
                seen['command'] = command
            row = json.loads((folder / 'input.jsonl').read_text(encoding='utf-8').splitlines()[0])
            (folder / 'predictions.jsonl').write_text(
                json.dumps({'id': row['id'], 'html': html, 'status': status}) + '\n', encoding='utf-8')
            return 0
        return fake

    def test_valid_prediction_completes_the_record(self):
        seen = {}
        self.service._run_inference = self.stub('<table><tr><td>42</td></tr></table>', seen=seen)
        result = self.service.run(self.image)
        self.assertTrue(result['html_valid'])
        self.assertTrue(Path(result['html_file']).is_file())
        folder = Path(result['output_directory'])
        self.assertTrue((folder / 'request.json').is_file())
        self.assertTrue((folder / 'result.json').is_file())
        self.assertEqual((folder / 'raw-output.txt').read_text(encoding='utf-8'),
                         '<table><tr><td>42</td></tr></table>')
        request = json.loads((folder / 'request.json').read_text(encoding='utf-8'))
        self.assertEqual(request['device'], 'cpu')
        self.assertEqual(request['min_pixels'], 65536)
        self.assertEqual(seen['env']['PYTHONPATH'].split(os.pathsep)[0], str(paths.PROJECT_ROOT))
        self.assertNotIn('.inference-deps', seen['env']['PYTHONPATH'])

    def test_truncated_output_keeps_failure_and_exports_nothing(self):
        self.service._run_inference = self.stub('<table><tr><td>42</td></tr></table>', status='truncated')
        result = self.service.run(self.image)
        self.assertFalse(result['html_valid'])
        self.assertIsNone(result['html_file'])
        self.assertFalse((Path(result['output_directory']) / 'table.html').exists())

    def test_invalid_html_is_not_exported(self):
        self.service._run_inference = self.stub('<table><tr><td>42</td></table>')
        result = self.service.run(self.image)
        self.assertFalse(result['html_valid'])
        self.assertIn('error_detail', result)
        self.assertFalse((Path(result['output_directory']) / 'table.html').exists())

    def test_wrong_adapter_hash_is_rejected_before_inference(self):
        service = DemoService(output=self.output, model=self.model, adapter=self.adapter,
                              python=Path(sys.executable), adapter_sha='0' * 64,
                              base_revision='test-revision')
        with self.assertRaisesRegex(ValueError, 'Q1'):
            service.run(self.image)

    def test_worker_timeout_is_recorded(self):
        def timeout(command, folder, env):
            raise subprocess.TimeoutExpired(command, 1)

        self.service._run_inference = timeout
        with self.assertRaises(subprocess.TimeoutExpired):
            self.service.run(self.image)
        failures = list((self.output / 'requests').glob('*/failure.json'))
        self.assertEqual(len(failures), 1)
        self.assertIn('error', json.loads(failures[0].read_text(encoding='utf-8')))

    def test_extra_pythonpath_is_forwarded_to_the_worker(self):
        extra = self.root / 'out-of-tree-deps'
        service = DemoService(output=self.output, model=self.model, adapter=self.adapter,
                              python=Path(sys.executable), adapter_sha=self.digest,
                              base_revision='test-revision', extra_pythonpath=str(extra))
        seen = {}
        service._run_inference = self.stub('<table><tr><td>42</td></tr></table>', seen=seen)
        service.run(self.image)
        pythonpath = seen['env']['PYTHONPATH'].split(os.pathsep)
        self.assertEqual(pythonpath[0], str(paths.PROJECT_ROOT))
        self.assertIn(str(extra.resolve()), pythonpath)

    def test_missing_weight_is_reported_before_launch(self):
        (self.model / 'model.safetensors').unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'download_model'):
            self.service.run(self.image)


if __name__ == '__main__':
    unittest.main()
