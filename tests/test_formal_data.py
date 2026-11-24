"""Checks for full-source sampling, archive paths and cross-paper quarantine."""

import contextlib
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from table_vlm.data import _key
from table_vlm.formal_data import safe_member, select_candidates, choose_subset, extract_annotations
from table_vlm.io_utils import read_jsonl, write_jsonl
from table_vlm.length_audit import distribution
from table_vlm.merge_paddle import merge
from table_vlm.training_view import retained_ids
from table_vlm.evaluate import numeric_tokens
from table_vlm.io_utils import sha256_file, write_json


class FormalDataTest(unittest.TestCase):
    def setUp(self) -> None:
        root = Path(__file__).resolve().parents[1]/'.cache/tests'
        root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=root)
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_archive_paths_links_and_devices_are_rejected(self) -> None:
        for name in ['../outside.png', '/outside.png', 'C:/file.png', 'pubtabnet\\file.png']:
            with self.assertRaises(ValueError):
                safe_member(tarfile.TarInfo(name))
        for kind in [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.CHRTYPE]:
            item = tarfile.TarInfo('pubtabnet/train/PMC1_001.png')
            item.type = kind
            with self.assertRaises(ValueError):
                safe_member(item)
        self.assertEqual(safe_member(tarfile.TarInfo('pubtabnet/train/PMC1_001.png')).name, 'PMC1_001.png')

    def test_archive_digest_is_required_before_any_extraction(self) -> None:
        archive = self.root/'unverified.tar.gz'
        archive.write_bytes(b'unverified')
        with self.assertRaises(ValueError):
            extract_annotations(archive, self.root/'output', {'bytes': 10, 'lfs_sha256': '0'*64})
        self.assertFalse((self.root/'output').exists())

    def test_archive_inspection_extracts_annotations_but_not_source_images(self) -> None:
        archive = self.root/'fixture.tar.gz'
        with tarfile.open(archive, 'w:gz') as tar:
            for name, content in [('pubtabnet/PubTabNet_2.0.0.jsonl', b'{"split":"train"}\n'),
                                  ('pubtabnet/test/PMC1_001.png', b'not inspected as an image')]:
                info = tarfile.TarInfo(name)
                info.size = len(content)
                tar.addfile(info, io.BytesIO(content))
        path = extract_annotations(archive, self.root/'output', {'bytes': archive.stat().st_size, 'lfs_sha256': sha256_file(archive)})
        self.assertEqual(path.read_bytes(), b'{"split":"train"}\n')
        self.assertFalse(list((self.root/'output').rglob('*.png')))
        receipt = json.loads((self.root/'output/raw/archive-inspection.json').read_text())
        self.assertEqual(receipt['file_counts']['.png'], 1)

    def test_seeded_selection_is_order_independent_and_quarantines_cross_source_papers(self) -> None:
        rows = [{'filename': f'PMC{n}_001.png', 'split': 'train' if n < 60 else 'val'} for n in range(100)]
        rows += [{'filename': 'PMC0_002.png', 'split': 'val'}]
        targets = {'train': 3, 'dev': 2, 'test': 2}
        outputs = []
        for label, source_rows in [('a', rows), ('b', list(reversed(rows)))]:
            root = self.root/label
            source = root/'source.jsonl'
            write_jsonl(source, source_rows)
            selected = select_candidates(source, root, targets, 42)
            outputs.append([r['filename'] for r in selected])
            receipt = json.loads((root/'raw/candidate-selection.json').read_text())
            self.assertIn('PMC0', receipt['cross_source_papers_quarantined'])
            self.assertFalse(any(r['filename'].startswith('PMC0_') for r in selected))
            train = [r for r in selected if r['split'] == 'train']
            expected = sorted((r for r in rows if r['split'] == 'train' and not r['filename'].startswith('PMC0_')), key=lambda r: _key(42, r['filename']))[:9]
            self.assertEqual(expected, train)
        self.assertEqual(*outputs)

    def test_candidate_selection_receipt_rejects_changed_source(self) -> None:
        source = self.root/'source.jsonl'
        write_jsonl(source, [{'filename': 'PMC1_001.png', 'split': 'train'}])
        select_candidates(source, self.root, {'train': 1, 'dev': 1, 'test': 1}, 42)
        source.write_text(source.read_text()+'\n', encoding='utf-8')
        with self.assertRaises(ValueError):
            select_candidates(source, self.root, {'train': 1, 'dev': 1, 'test': 1}, 42)

    def test_length_distribution_keeps_maximum_and_tail(self) -> None:
        result = distribution([100, 2, 300, 4, 9000])
        self.assertEqual(result['p100'], 9000)
        self.assertEqual(result['p50'], 100)
        self.assertEqual(distribution([]), {})

    def test_unicode_minus_is_not_lost_in_numeric_diagnostics(self) -> None:
        minus = '<table><tr><td>−1.4 ± 1.2</td></tr></table>'
        positive = '<table><tr><td>1.4 ± 1.2</td></tr></table>'
        ascii_minus = '<table><tr><td>-1.4 ± 1.2</td></tr></table>'
        self.assertNotEqual(numeric_tokens(minus), numeric_tokens(positive))
        self.assertEqual(numeric_tokens(minus), numeric_tokens(ascii_minus))
        self.assertNotEqual(numeric_tokens('<table><tr><td>.615</td></tr></table>'),
                            numeric_tokens('<table><tr><td>615</td></tr></table>'))

    def test_training_length_guard_uses_union_and_leaves_heldout_untouched(self) -> None:
        clean = [{'id': 'a', 'split': 'train', 'multimodal_tokens': 100},
                 {'id': 'b', 'split': 'train', 'multimodal_tokens': 4000},
                 {'id': 'heldout', 'split': 'dev', 'multimodal_tokens': 9000}]
        augmented = [dict(r) for r in clean]
        augmented[0]['multimodal_tokens'] = 4500
        allowed, excluded = retained_ids(clean, augmented, 4032)
        self.assertEqual(allowed, {'b'})
        self.assertEqual([r['id'] for r in excluded], ['a'])
        self.assertEqual(clean[-1]['multimodal_tokens'], 9000)
        with self.assertRaises(ValueError):
            retained_ids(clean, augmented[:-1], 4032)

    def test_decoded_pixel_duplicates_are_quarantined_despite_different_file_bytes(self) -> None:
        rows = []
        # Different source splits; the first two dev candidates have identical pixels but different PNG metadata.
        from PIL.PngImagePlugin import PngInfo
        for n in range(100, 160):
            split = 'train' if n < 120 else 'val'
            row = {'filename': f'PMC{n}_001.png', 'split': split, 'html': {
                'structure': {'tokens': ['<tbody><tr><td></td></tr></tbody>']}, 'cells': [{'tokens': ['A']}]}}
            path = self.root/'images'/split/row['filename']
            path.parent.mkdir(parents=True, exist_ok=True)
            image = Image.new('RGB', (40+n, 25), (n, 30, 40))
            image.save(path)
            rows.append(row)
        dev = sorted([r for r in rows if r['split'] == 'val' and int(_key(42, r['filename'].split('_')[0])[:8], 16)%2==0], key=lambda r: _key(42, r['filename']))
        original = self.root/'images/val'/dev[0]['filename']
        with Image.open(original) as image:
            info = PngInfo()
            info.add_text('note', 'same pixels, different bytes')
            image.save(self.root/'images/val'/dev[1]['filename'], pnginfo=info)
        self.assertNotEqual(sha256_file(original), sha256_file(self.root/'images/val'/dev[1]['filename']))
        with contextlib.redirect_stdout(io.StringIO()):
            choose_subset(rows, self.root/'images', self.root, {'train': 1, 'dev': 2, 'test': 1}, 42)
        rejected = read_jsonl(self.root/'reports/data/rejected-candidates.jsonl')
        self.assertTrue(any(r['id'] == Path(dev[1]['filename']).stem and r['reason']=='exact_bytes_or_decoded_pixels_duplicate' for r in rejected))

    def test_cpu_merge_rejects_missing_duplicate_and_wrong_device_predictions(self) -> None:
        manifest = self.root/'dev.jsonl'
        write_jsonl(manifest, [{'id': 'a'}, {'id': 'b'}])
        paths = []
        for n, sample_id in enumerate(['a', 'b']):
            path = self.root/f'shard-{n}.jsonl'
            write_jsonl(path, [{'id': sample_id, 'status': 'generation_error', 'html': ''}])
            write_json(path.with_suffix('.run.json'), {'manifest_sha256': sha256_file(manifest),
                'device': 'cpu', 'compiled_with_cuda': False, 'config_sha256': 'same', 'versions': {},
                'models': {}, 'cpu_threads': 4, 'shards': 2})
            paths.append(path)
        with self.assertRaises(ValueError):
            merge(manifest, paths[:1], self.root/'missing.jsonl')
        with self.assertRaises(ValueError):
            merge(manifest, [paths[0], paths[0]], self.root/'duplicate.jsonl')
        merge(manifest, paths, self.root/'complete.jsonl')
        self.assertEqual(len(read_jsonl(self.root/'complete.jsonl')), 2)
        receipt = json.loads(paths[0].with_suffix('.run.json').read_text())
        receipt['device'] = 'gpu'
        paths[0].with_suffix('.run.json').write_text(json.dumps(receipt))
        with self.assertRaises(ValueError):
            merge(manifest, paths, self.root/'wrong-device.jsonl')


if __name__ == '__main__':
    unittest.main()
