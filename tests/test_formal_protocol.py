"""A successful old gate must not authorize changed data or incomplete labels."""

import json
from pathlib import Path
import tempfile
import unittest

from table_vlm.formal_protocol import GATE, verify_gate
from table_vlm.io_utils import sha256_file, write_json, write_jsonl


class FormalGateTest(unittest.TestCase):
    def setUp(self):
        cache = Path(__file__).resolve().parents[1]/'.cache/tests'
        cache.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=cache)
        self.root = Path(self.temp.name)
        config = self.root/'configs/lora_24gb.yaml'
        config.parent.mkdir()
        config.write_text('num_train_epochs: 2.0\ncutoff_len: 4096\npacking: false\nimage_max_pixels: 196608\n')
        receipt = self.root/'models/Qwen3-VL-2B-Instruct/source_receipt.json'
        write_json(receipt, {})
        gate = self.root/GATE
        write_json(gate/'exit-status.json', {'exit_code': 0})
        write_json(gate/'ready.json', {'passed': True})
        for profile in ['formal-clean', 'formal-aug']:
            summary = {'passed': True, 'failures': 0, 'preflight_version': 'lf-sft-v3-nfc-exact-tokens',
                       'formal_config_sha256': sha256_file(config),
                       'model_receipt_sha256': sha256_file(receipt), 'splits': {}}
            encoded = []
            for split, count in [('train', 2990), ('dev', 150)]:
                rows = [{'id': f'{split}{i}'} for i in range(count)]
                manifest = self.root/'data/prepared'/profile/f'{split}.jsonl'
                write_jsonl(manifest, rows)
                summary['splits'][split] = {'manifest_sha256': sha256_file(manifest)}
                encoded.extend({'id': r['id'], 'split': split, 'tokens': 100,
                                'complete_target': True, 'exact_answer_tokens': True} for r in rows)
            write_jsonl(gate/profile/'samples.jsonl', encoded)
            write_jsonl(gate/profile/'failures.jsonl', [])
            write_json(gate/profile/'summary.json', summary)

    def tearDown(self):
        self.temp.cleanup()

    def test_valid_gate_then_changed_manifest_is_rejected(self):
        self.assertTrue(verify_gate(self.root)['passed'])
        p = self.root/'data/prepared/formal-clean/train.jsonl'
        with p.open('a') as f:
            f.write('\n')
        with self.assertRaisesRegex(ValueError, 'Manifest changed'):
            verify_gate(self.root)

    def test_incomplete_supervised_answer_is_rejected(self):
        p = self.root/GATE/'formal-clean/samples.jsonl'
        records = [json.loads(line) for line in p.read_text().splitlines()]
        records[0]['exact_answer_tokens'] = False
        p.write_text(''.join(json.dumps(r)+'\n' for r in records))
        with self.assertRaisesRegex(ValueError, 'invalid encoding record'):
            verify_gate(self.root)

    def test_changed_configuration_is_rejected(self):
        p = self.root/'configs/lora_24gb.yaml'
        with p.open('a') as f:
            f.write('max_steps: 8\n')
        with self.assertRaisesRegex(ValueError, 'Stale or failed'):
            verify_gate(self.root)
