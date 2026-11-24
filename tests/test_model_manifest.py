"""Model receipt integrity checks without loading a model or using a network."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from table_vlm.download_model import download_from_manifest


class ModelManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        root = Path(__file__).resolve().parents[1]/'.cache/tests'
        root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=root)
        self.root = Path(self.temp.name)
        self.output = self.root/'model'
        self.output.mkdir()
        files = []
        for name, data in [('config.json', b'{}'), ('model.safetensors', b'fixture, not real weights')]:
            (self.output/name).write_bytes(data)
            files.append({'path': name, 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)})
        self.manifest = self.root/'manifest.json'
        self.record = {'repo': 'Qwen/Qwen3-VL-2B-Instruct', 'revision': '0'*40, 'files': files}
        self.manifest.write_text(json.dumps(self.record))

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_existing_files_are_verified_and_receipt_is_idempotent(self) -> None:
        for _ in range(2):
            result = download_from_manifest(self.output, self.manifest, 'modelscope')
            self.assertTrue(result['hashes_verified'])
            self.assertEqual(result['metadata_network_requests'], 0)

    def test_corrupted_file_cannot_produce_receipt(self) -> None:
        (self.output/'model.safetensors').write_bytes(b'bad')
        with self.assertRaises(ValueError):
            download_from_manifest(self.output, self.manifest, 'modelscope')
        self.assertFalse((self.output/'source_receipt.json').exists())

    def test_revision_mismatch_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            download_from_manifest(self.output, self.manifest, 'modelscope', '1'*40)

    def test_escaping_path_is_rejected(self) -> None:
        self.record['files'][0]['path'] = '../outside.json'
        self.manifest.write_text(json.dumps(self.record))
        with self.assertRaises(ValueError):
            download_from_manifest(self.output, self.manifest, 'modelscope')

