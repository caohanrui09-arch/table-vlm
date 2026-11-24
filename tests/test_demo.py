"""Prediction previews must preserve failures and never execute model HTML."""

from pathlib import Path
import tempfile
import unittest

from table_vlm.demo import present_prediction


class DemoOutputTest(unittest.TestCase):
    def setUp(self):
        cache = Path(__file__).resolve().parents[1]/'.cache/tests'
        cache.mkdir(parents=True,exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=cache)
        self.folder = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_valid_export_strips_metadata_and_is_sandboxed(self):
        raw = '<table data-bbox="0,0,1,1" style="background:url(https://example.com)"><tr><td colspan="2">42</td></tr></table>'
        result = present_prediction({'status':'ok','html':raw},self.folder)
        self.assertTrue(result['html_valid'])
        saved = Path(result['html_file']).read_text(encoding='utf-8')
        self.assertIn('colspan="2"',saved)
        self.assertNotIn('data-bbox',saved)
        self.assertNotIn('https://',saved)
        self.assertIn('Content-Security-Policy',saved)
        self.assertIn('sandbox=""',result['preview'])
        self.assertEqual((self.folder/'raw-output.txt').read_text(encoding='utf-8'),raw)

    def test_invalid_model_html_is_not_rendered_or_exported(self):
        raw = '<table onclick="alert(1)"><tr><td>42</td></tr></table>'
        result = present_prediction({'status':'ok','html':raw},self.folder)
        self.assertFalse(result['html_valid'])
        self.assertEqual(result['preview'],'')
        self.assertIsNone(result['html_file'])
        self.assertFalse((self.folder/'table.html').exists())

    def test_truncation_is_preserved_even_when_partial_html_balances(self):
        raw = '<table><tr><td>42</td></tr></table>'
        result = present_prediction({'status':'truncated','html':raw},self.folder)
        self.assertFalse(result['html_valid'])
        self.assertEqual(result['raw'],raw)
        self.assertIsNone(result['html_file'])
