"""Verify prepared manifests, actual images and LlamaFactory exports on CPU."""

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from .html_utils import canonical_table, table_attributes
from .io_utils import read_jsonl, sha256_file, unique_by_id, write_json
from .prompts import PROMPT


def verify_dataset(data: Path, reference: Path | None = None) -> dict:
    data = data.resolve()
    receipt = json.loads((data/'audit.json').read_text(encoding='utf-8'))
    export_info = json.loads((data/'lf/dataset_info.json').read_text(encoding='utf-8'))
    papers: dict[str, str] = {}
    image_hashes: dict[str, str] = {}
    all_ids: set[str] = set()
    summary = {}
    split_records = {}
    for split in ['train', 'dev', 'test']:
        path = data/f'{split}.jsonl'
        if sha256_file(path) != receipt['manifest_sha256'][split]:
            raise ValueError(f'{split} manifest has changed since preparation')
        rows = read_jsonl(path)
        unique_by_id(rows)
        if len(rows) != receipt['split_counts'][split] or not rows:
            raise ValueError(f'{split} count differs from audit or is empty')
        if sorted({r['paper_id'] for r in rows}) != receipt['paper_ids'][split]:
            raise ValueError(f'{split} paper IDs differ from audit')
        split_records[split] = rows
        expected_registration = {'file_name': f'{split}.json', 'formatting': 'sharegpt',
                                 'columns': {'messages': 'conversations', 'images': 'images'}}
        if export_info.get(f'table_{split}') != expected_registration:
            raise ValueError(f'Incorrect LlamaFactory registration: {split}')
        lf_path = data/'lf'/f'{split}.json'
        exported = json.loads(lf_path.read_text(encoding='utf-8'))
        if len(exported) != len(rows):
            raise ValueError(f'LlamaFactory sample count differs: {split}')
        sizes, cells, row_counts, operations = [], [], [], Counter()
        for row, lf_row in zip(rows, exported, strict=True):
            if row['id'] in all_ids or row['split'] != split:
                raise ValueError(f'Duplicate ID or wrong split: {row["id"]}')
            all_ids.add(row['id'])
            group = row['paper_id']
            if group in papers and papers[group] != split:
                raise ValueError(f'Paper leaks between splits: {group}')
            papers[group] = split
            target = canonical_table(row['html'])
            expected_lf = {'conversations': [
                {'from': 'human', 'value': '<image>\n'+PROMPT},
                {'from': 'gpt', 'value': target.replace('<html><body>', '').replace('</body></html>', '')}],
                'images': [row['image']]}
            if lf_row != expected_lf:
                raise ValueError(f'LlamaFactory image/prompt/target differs from manifest: {row["id"]}')
            image_path = (data/row['image']).resolve()
            if not image_path.is_relative_to(data):
                raise ValueError('Image path escapes data directory')
            actual_hash = sha256_file(image_path)
            if actual_hash != row['image_sha256']:
                raise ValueError(f'Image changed: {row["id"]}')
            if actual_hash in image_hashes and image_hashes[actual_hash] != split:
                raise ValueError('Identical image leaks between splits')
            image_hashes[actual_hash] = split
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                if image.size != (row['width'], row['height']):
                    raise ValueError(f'Image dimensions differ: {row["id"]}')
                sizes.append(image.size)
            attrs = table_attributes(target)
            if any(row['attributes'].get(k) != v for k, v in attrs.items()):
                raise ValueError(f'Stale table attributes: {row["id"]}')
            operations[row['transform']['operation']] += 1
            if split != 'train' and (row['transform'] != {'operation': 'clean'} or
                                     actual_hash != row['source_image_sha256']):
                raise ValueError('Held-out image has been altered')
            if row['transform']['operation'] == 'clean' and actual_hash != row['source_image_sha256']:
                raise ValueError('Image marked clean differs from source')
            cells.append(attrs['cells'])
            row_counts.append(attrs['rows'])
        summary[split] = {'samples': len(rows), 'papers': len({r['paper_id'] for r in rows}),
                          'rows_range': [min(row_counts), max(row_counts)],
                          'cells_range': [min(cells), max(cells)],
                          'image_width_range': [min(w for w, h in sizes), max(w for w, h in sizes)],
                          'image_height_range': [min(h for w, h in sizes), max(h for w, h in sizes)],
                          'merged_tables': sum(r['attributes']['merged'] for r in rows),
                          'transforms': dict(operations), 'lf_sha256': sha256_file(lf_path)}
    comparisons = {}
    if reference:
        reference = reference.resolve()
        reference_receipt = json.loads((reference/'audit.json').read_text(encoding='utf-8'))
        if reference_receipt['augmentation']:
            raise ValueError('The reference must be a clean dataset')
        if receipt['seed'] != reference_receipt['seed']:
            raise ValueError('Augmentation comparison changed split seed')
        for split, rows in split_records.items():
            original = read_jsonl(reference/f'{split}.jsonl')
            if len(rows) != len(original):
                raise ValueError('Augmentation changed sample exposure counts')
            for a, b in zip(rows, original, strict=True):
                for key in ['id', 'paper_id', 'html', 'source_split', 'split', 'source_image_sha256']:
                    if a[key] != b[key]:
                        raise ValueError(f'Augmentation changed sample order/identity/target: {key}')
                if split != 'train' and a != b:
                    raise ValueError('Augmentation changed held-out data')
            comparisons[split] = {'same_order_targets_and_count': True,
                                  'manifest_byte_identical': sha256_file(data/f'{split}.jsonl') == sha256_file(reference/f'{split}.jsonl')}
    return {'status': 'passed', 'mode': receipt['mode'], 'samples': len(all_ids), 'split_summary': summary,
            'checks': ['manifest and audit binding', 'all LlamaFactory records match manifest', 'image hashes and dimensions',
                       'paper and exact-image separation', 'held-out images unchanged'],
            'augmentation_reference': comparisons, 'near_duplicate_check': receipt.get('near_duplicate_check', 'not performed'),
            'note': 'Data integrity verification only, not a model accuracy result.'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = verify_dataset(args.data, args.reference)
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
