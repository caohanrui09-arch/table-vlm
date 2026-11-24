"""Preflight checks must fail with actionable messages, not tracebacks."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from table_vlm import doctor, paths


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class DoctorTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def make_model(self, *, weight: bytes = b'model', revision: str = paths.BASE_REVISION,
                   with_receipt: bool = True, with_weight: bool = True) -> tuple[Path, Path]:
        model = self.root / 'model'
        model.mkdir()
        files = []
        for name in paths.MODEL_METADATA:
            payload = f'{name}\n'.encode()
            (model / name).write_bytes(payload)
            files.append({'path': name, 'bytes': len(payload), 'sha256': sha(payload)})
        if with_weight:
            (model / paths.MODEL_WEIGHT).write_bytes(weight)
            files.append({'path': paths.MODEL_WEIGHT, 'bytes': len(weight), 'sha256': sha(weight)})
        if with_receipt:
            (model / 'source_receipt.json').write_text(json.dumps({'revision': revision, 'files': files}),
                                                       encoding='utf-8')
        manifest = self.root / 'manifest.json'
        manifest.write_text(json.dumps({
            'repo': 'Qwen/Qwen3-VL-2B-Instruct', 'revision': revision,
            'files': [{'path': paths.MODEL_WEIGHT, 'bytes': len(weight), 'sha256': sha(weight)}],
        }), encoding='utf-8')
        return model, manifest

    def make_adapter(self, payload: bytes = b'adapter') -> tuple[Path, str]:
        adapter = self.root / 'adapter'
        adapter.mkdir()
        (adapter / 'adapter_model.safetensors').write_bytes(payload)
        (adapter / 'adapter_config.json').write_text(json.dumps({
            'r': 8, 'lora_alpha': 16, 'target_modules': ['v_proj', 'q_proj'], 'peft_type': 'LORA',
        }), encoding='utf-8')
        return adapter, sha(payload)

    def test_complete_model_and_adapter_pass(self):
        model, manifest = self.make_model()
        adapter, digest = self.make_adapter()
        self.assertEqual(doctor.check_model(model, manifest)['status'], doctor.OK)
        self.assertEqual(doctor.check_model(model, manifest, deep=True)['status'], doctor.OK)
        self.assertEqual(doctor.check_adapter(adapter, digest)['status'], doctor.OK)

    def test_missing_weight_tells_user_to_download(self):
        model, manifest = self.make_model(with_weight=False)
        result = doctor.check_model(model, manifest)
        self.assertEqual(result['status'], doctor.FAIL)
        self.assertIn('download_model', result['hint'])

    def test_wrong_base_revision_fails(self):
        model, manifest = self.make_model(revision='f' * 40)
        result = doctor.check_model(model, manifest)
        self.assertEqual(result['status'], doctor.FAIL)
        self.assertIn(paths.BASE_REVISION, result['detail'])

    def test_truncated_weight_against_manifest_fails(self):
        model, manifest = self.make_model(weight=b'short')
        manifest_data = json.loads(manifest.read_text(encoding='utf-8'))
        manifest_data['files'][0]['bytes'] = 999
        manifest.write_text(json.dumps(manifest_data), encoding='utf-8')
        result = doctor.check_model(model, manifest)
        self.assertEqual(result['status'], doctor.FAIL)
        self.assertIn('manifest says 999', result['detail'])

    def test_adapter_hash_mismatch_fails(self):
        adapter, _ = self.make_adapter(b'one')
        result = doctor.check_adapter(adapter, sha(b'two'))
        self.assertEqual(result['status'], doctor.FAIL)
        self.assertIn('frozen Q1', result['detail'])

    def test_output_and_port_reports(self):
        output = self.root / 'demo'
        self.assertEqual(doctor.check_output(output)['status'], doctor.OK)
        self.assertEqual(doctor.check_port('127.0.0.1', 1)['status'], doctor.OK)

    def test_missing_runtime_interpreter_fails(self):
        result = doctor.check_runtime(self.root / 'no-such-python.exe')
        self.assertEqual(result['status'], doctor.FAIL)
        self.assertIn('not found', result['detail'])

    def test_run_doctor_ready_and_report(self):
        model, manifest = self.make_model()
        report = doctor.run_doctor(model=model, adapter=paths.PROJECT_ROOT / 'artifacts/q1-adapter',
                                   manifest=manifest, gpu_python=Path('python'),
                                   output=self.root / 'demo', skip_runtime=True, skip_port=True)
        self.assertTrue(report['ok'])
        self.assertIn('READY', doctor.format_report(report))


if __name__ == '__main__':
    unittest.main()
