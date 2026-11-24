"""Final test cannot silently use changed weights, data or unregistered splits."""

import json
from pathlib import Path
import tempfile
import unittest

from table_vlm.final_test_protocol import verify_files
from table_vlm.predict_paddle_cpu import authorize_split
from table_vlm.io_utils import sha256_file,write_json,write_jsonl


class FinalTestProtocolTest(unittest.TestCase):
    def setUp(self):
        cache=Path(__file__).resolve().parents[1]/'.cache/tests';cache.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=cache);self.root=Path(self.temp.name)
        self.model=self.root/'adapter.safetensors';self.model.write_bytes(b'fixture')
        self.rows=[{'id':'example','split':'test'}]
        self.manifest=self.root/'test.jsonl';write_jsonl(self.manifest,self.rows)
        self.config=self.root/'paddle.yaml';self.config.write_text('fixture')
        self.freeze=self.root/'freeze.json'
        write_json(self.freeze,{'selected_model':'q1','test_samples':1,
            'test_manifest_sha256':sha256_file(self.manifest),'paddle':{'config_sha256':sha256_file(self.config)}})

    def tearDown(self):self.temp.cleanup()

    def test_changed_adapter_is_rejected(self):
        bound={'adapter.safetensors':sha256_file(self.model)}
        verify_files(self.root,bound)
        self.model.write_bytes(b'different model')
        with self.assertRaisesRegex(ValueError,'Frozen file changed'):
            verify_files(self.root,bound)

    def test_paths_outside_the_project_are_rejected(self):
        with self.assertRaises(ValueError):verify_files(self.root,{'../outside':'x'})

    def test_test_requires_explicit_registered_freeze(self):
        with self.assertRaises(ValueError):authorize_split(self.rows,self.manifest,self.config,None)
        self.assertTrue(authorize_split(self.rows,self.manifest,self.config,self.freeze))

    def test_changed_test_manifest_is_rejected(self):
        with self.manifest.open('a') as f:f.write('\n')
        with self.assertRaises(ValueError):authorize_split(self.rows,self.manifest,self.config,self.freeze)

    def test_dev_still_works_without_test_authorization(self):
        self.assertFalse(authorize_split([{'split':'dev'}],self.manifest,self.config,None))
        with self.assertRaises(ValueError):
            authorize_split([{'split':'dev'}],self.manifest,self.config,self.freeze)
