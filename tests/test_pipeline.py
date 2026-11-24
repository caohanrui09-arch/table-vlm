"""Correctness checks for leakage, annotation assembly and honest evaluation."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from table_vlm.data import allocate, augment_image, document_id, prepare
from table_vlm.evaluate import evaluate, load_teds
from table_vlm.html_utils import InvalidTable, annotation_html, canonical_table, cell_texts
from table_vlm.io_utils import read_jsonl, unique_by_id, write_jsonl
from table_vlm.audit_dataset import verify_dataset


ROOT = Path(__file__).resolve().parents[1]
SIMPLE = '<table><tbody><tr><td>12</td><td>name</td></tr></tbody></table>'


class PipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        base = ROOT/'.cache/tests'
        base.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=base)
        self.root = Path(self.temp.name).resolve()

    def tearDown(self) -> None:
        self.assertTrue(self.root.is_relative_to(ROOT.resolve()))
        self.temp.cleanup()

    def make_sources(self) -> tuple[Path, Path]:
        images = self.root/'images'
        images.mkdir()
        rows = []
        for i in range(10):
            name = f'PMC{1000+i}_001_00.png'
            Image.new('RGB', (20+i, 15), (20*i, 10, 200)).save(images/name)
            rows.append({'filename': name, 'split': 'train', 'html': {
                'structure': {'tokens': ['<tbody><tr><td></td></tr></tbody>']},
                'cells': [{'tokens': list(str(i))}]}})
        path = self.root/'source.jsonl'
        write_jsonl(path, rows)
        return path, images

    def test_annotation_keeps_spans_inline_markup_and_literal_symbols(self) -> None:
        row = {'html': {'structure': {'tokens': ['<tbody><tr><td', ' colspan="2"', '>', '</td></tr></tbody>']},
                        'cells': [{'tokens': ['<b>', 'A', '&', '<', '3', '</b>']}]}}
        result = annotation_html(row)
        self.assertIn('colspan="2"', result)
        self.assertIn('<b>A&amp;&lt;3</b>', result)
        self.assertEqual(cell_texts(result), ['A&<3'])

    def test_annotation_cell_mismatch_is_fatal(self) -> None:
        with self.assertRaises(InvalidTable):
            annotation_html({'html': {'structure': {'tokens': ['<tr><td></td></tr>']}, 'cells': []}})

    def test_truncated_and_unsafe_html_are_not_repaired(self) -> None:
        for bad in ['<table><tr><td>1', '<table><tr><td>1</tr></td></table>',
                    '<table><tr><td><script>alert(1)</script></td></tr></table>',
                    '<table><td>1</td></table>', '<table><tr><td rowspan="0">1</td></tr></table>']:
            with self.subTest(bad=bad), self.assertRaises(InvalidTable):
                canonical_table(bad)

    def test_markdown_fence_and_th_normalize_equally(self) -> None:
        a = canonical_table('```html\n<table><tr><th>A</th></tr></table>\n```')
        b = canonical_table('<table><tr><td>A</td></tr></table>')
        self.assertEqual(a, b)

    def test_teds_text_and_structure_are_different_metrics(self) -> None:
        a = canonical_table(SIMPLE)
        b = canonical_table(SIMPLE.replace('12', '99'))
        TEDS = load_teds()
        self.assertEqual(TEDS().evaluate(a, a), 1.0)
        self.assertLess(TEDS().evaluate(b, a), 1.0)
        self.assertEqual(TEDS(structure_only=True).evaluate(b, a), 1.0)

    def test_path_traversal_and_unknown_paper_id_are_rejected(self) -> None:
        for name in ['../PMC123_1.png', 'C:\\x\\PMC1.png', 'arbitrary.png']:
            with self.subTest(name=name), self.assertRaises(ValueError):
                document_id(name)

    def test_paper_cannot_cross_original_source_splits(self) -> None:
        with self.assertRaises(ValueError):
            allocate([{'filename': 'PMC123_001.png', 'split': 'train'},
                      {'filename': 'PMC123_002.png', 'split': 'val'}], 'official', 42)

    def test_same_paper_multiple_tables_share_one_split(self) -> None:
        rows = [{'filename': f'PMC{n}_{i:03}.png', 'split': 'train'} for n in range(5) for i in range(2)]
        result = allocate(rows, 'smoke', 42)
        self.assertEqual(len(result), 5)
        self.assertEqual(result, allocate(list(reversed(rows)), 'smoke', 42))
        self.assertEqual(set(result.values()), {'train', 'dev', 'test'})

    def test_duplicate_prediction_ids_are_fatal(self) -> None:
        with self.assertRaises(ValueError):
            unique_by_id([{'id': 'x'}, {'id': 'x'}])

    def test_missing_invalid_and_truncated_predictions_stay_in_denominator(self) -> None:
        truth = self.root/'truth.jsonl'
        pred = self.root/'pred.jsonl'
        write_jsonl(truth, [{'id': str(i), 'html': SIMPLE} for i in range(4)])
        write_jsonl(pred, [{'id': '0', 'html': SIMPLE}, {'id': '1', 'html': '<table>'},
                          {'id': '2', 'html': SIMPLE, 'status': 'truncated'}])
        with contextlib.redirect_stdout(io.StringIO()):
            report = evaluate(truth, pred, self.root/'report')
        self.assertEqual(report['samples'], 4)
        self.assertEqual(report['teds'], .25)
        self.assertEqual(report['valid_html_rate'], .25)
        self.assertEqual(report['missing_predictions'], 1)
        self.assertFalse(report['quality_claim_allowed'])

    def test_wrong_split_prediction_is_fatal(self) -> None:
        truth, pred = self.root/'truth.jsonl', self.root/'pred.jsonl'
        write_jsonl(truth, [{'id': 'x', 'html': SIMPLE}])
        write_jsonl(pred, [{'id': 'wrong', 'html': SIMPLE}])
        with self.assertRaises(ValueError):
            evaluate(truth, pred, self.root/'report')

    def test_exports_have_disjoint_papers_and_augmentation_preserves_holdout(self) -> None:
        annotations, images = self.make_sources()
        with contextlib.redirect_stdout(io.StringIO()):
            receipt = prepare(annotations, images, self.root/'clean')
            prepare(annotations, images, self.root/'aug', augment=True)
        papers = [set(receipt['paper_ids'][s]) for s in ['train', 'dev', 'test']]
        self.assertFalse(papers[0]&papers[1] or papers[0]&papers[2] or papers[1]&papers[2])
        for split in ['dev', 'test']:
            clean = read_jsonl(self.root/f'clean/{split}.jsonl')
            aug = read_jsonl(self.root/f'aug/{split}.jsonl')
            self.assertEqual(clean, aug)
        lf = json.loads((self.root/'clean/lf/train.json').read_text(encoding='utf-8'))
        self.assertIn('<image>', lf[0]['conversations'][0]['value'])
        self.assertEqual(lf[0]['conversations'][1]['from'], 'gpt')

    def test_official_mode_does_not_accept_train_only_samples(self) -> None:
        annotations, images = self.make_sources()
        with self.assertRaises(ValueError):
            prepare(annotations, images, self.root/'official', mode='official')

    def test_identical_images_cannot_cross_splits(self) -> None:
        annotations, images = self.make_sources()
        data = (images/'PMC1000_001_00.png').read_bytes()
        for path in images.glob('*.png'):
            path.write_bytes(data)
        with self.assertRaises(ValueError):
            prepare(annotations, images, self.root/'duplicate')

    def test_augmentation_is_reproducible(self) -> None:
        image = Image.new('RGB', (50, 50), 'white')
        a, params_a = augment_image(image, 'sample', 42)
        b, params_b = augment_image(image, 'sample', 42)
        self.assertEqual(params_a, params_b)
        self.assertEqual(a.tobytes(), b.tobytes())

    def test_safe_presentation_attributes_do_not_change_teds(self) -> None:
        decorated = SIMPLE.replace('<table>', '<table style="color:red" border="1" class="example">')
        self.assertEqual(canonical_table(decorated), canonical_table(SIMPLE))
        with self.assertRaises(InvalidTable):
            canonical_table(SIMPLE.replace('<table>', '<table onclick="bad()">'))

    def test_fostered_text_and_invalid_inline_structure_are_rejected(self) -> None:
        for bad in ['<table>hidden text<tr><td>A</td></tr></table>',
                    '<table><tr><b><td>A</td></b></tr></table>',
                    '<table><tr><td rowspan="1" rowspan="2">A</td></tr></table>']:
            with self.subTest(bad=bad), self.assertRaises(InvalidTable):
                canonical_table(bad)

    def test_lf_export_tampering_is_detected(self) -> None:
        annotations, images = self.make_sources()
        with contextlib.redirect_stdout(io.StringIO()):
            prepare(annotations, images, self.root/'clean')
        self.assertEqual(verify_dataset(self.root/'clean')['status'], 'passed')
        path = self.root/'clean/lf/train.json'
        rows = json.loads(path.read_text(encoding='utf-8'))
        rows[0]['conversations'][1]['value'] = '<table><tr><td>wrong</td></tr></table>'
        path.write_text(json.dumps(rows), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'LlamaFactory'):
            verify_dataset(self.root/'clean')

    def test_augmentation_audit_preserves_clean_bytes_and_sample_exposure(self) -> None:
        annotations, images = self.make_sources()
        with contextlib.redirect_stdout(io.StringIO()):
            prepare(annotations, images, self.root/'clean')
            prepare(annotations, images, self.root/'aug', augment=True)
        result = verify_dataset(self.root/'aug', self.root/'clean')
        self.assertTrue(result['augmentation_reference']['dev']['manifest_byte_identical'])
        self.assertTrue(result['augmentation_reference']['train']['same_order_targets_and_count'])

    def test_changed_image_is_detected_after_manifest_is_frozen(self) -> None:
        annotations, images = self.make_sources()
        with contextlib.redirect_stdout(io.StringIO()):
            prepare(annotations, images, self.root/'clean')
        first = read_jsonl(self.root/'clean/dev.jsonl')[0]
        (self.root/'clean'/first['image']).write_bytes(b'not an image')
        with self.assertRaisesRegex(ValueError, 'Image changed'):
            verify_dataset(self.root/'clean')

    def test_bbox_metadata_does_not_change_structure_or_text(self) -> None:
        plain = '<table><tr><td colspan="2">42</td></tr></table>'
        with_bbox = '<table data-bbox="0,0,1000,1000"><tr><td data-bbox="1,2,3,4" colspan="2">42</td></tr></table>'
        self.assertEqual(canonical_table(plain), canonical_table(with_bbox))

    def test_bbox_does_not_allow_event_attributes_or_repair_truncation(self) -> None:
        for value in ['<table data-bbox="0,0,1,1" onclick="bad()"><tr><td>42</td></tr></table>',
                      '<table data-bbox="0,0,1,1"><tr><td>42</td>']:
            with self.assertRaises(InvalidTable):
                canonical_table(value)


if __name__ == '__main__':
    unittest.main()
