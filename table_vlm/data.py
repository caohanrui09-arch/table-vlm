"""Audit PubTabNet records and export the existing LlamaFactory ShareGPT schema."""

import argparse
import hashlib
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageFilter

from .html_utils import annotation_html, table_attributes
from .io_utils import iter_jsonl, sha256_file, write_json, write_jsonl
from .prompts import PROMPT


def document_id(filename: str) -> str:
    if Path(filename).name != filename or '/' in filename or '\\' in filename:
        raise ValueError(f'Unsafe filename: {filename}')
    match = re.match(r'^(PMC\d+)_', filename)
    if not match:
        raise ValueError(f'Cannot determine paper ID: {filename}')
    return match.group(1)


def _key(seed: int, value: str) -> str:
    return hashlib.sha256(f'{seed}:{value}'.encode()).hexdigest()


def allocate(rows: list[dict], mode: str, seed: int) -> dict[str, str]:
    source_sets: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        source_sets[document_id(row['filename'])].add(row['split'])
    if any(len(sources) > 1 for sources in source_sets.values()):
        raise ValueError('A paper crosses original source splits')
    groups = sorted(source_sets, key=lambda g: _key(seed, g))
    if mode == 'smoke':
        if len(groups) < 3:
            raise ValueError('Smoke split requires at least three papers')
        n_dev = max(1, int(len(groups) * .15))
        n_test = max(1, int(len(groups) * .15))
        result = {g: 'train' for g in groups}
        result.update({g: 'dev' for g in groups[:n_dev]})
        result.update({g: 'test' for g in groups[n_dev:n_dev+n_test]})
        return result
    result = {}
    for group in groups:
        source = next(iter(source_sets[group]))
        if source == 'train':
            result[group] = 'train'
        elif source == 'val':
            # Held-out validation papers become local dev/test, not official test.
            result[group] = 'dev' if int(_key(seed, group)[:8], 16) % 2 == 0 else 'test'
        else:
            raise ValueError('Official mode accepts labeled train/val only; do not mix official test protocols')
    return result


def augment_image(image: Image.Image, sample_id: str, seed: int) -> tuple[Image.Image, dict]:
    rng = random.Random(_key(seed, sample_id))
    operation = rng.choice(['clean', 'clean', 'blur', 'low_resolution', 'tilt'])
    if operation == 'blur':
        radius = round(rng.uniform(.35, .8), 3)
        return image.filter(ImageFilter.GaussianBlur(radius)), {'operation': operation, 'radius': radius}
    if operation == 'low_resolution':
        scale = round(rng.uniform(.55, .8), 3)
        small = image.resize((max(1, int(image.width*scale)), max(1, int(image.height*scale))), Image.Resampling.BICUBIC)
        return small.resize(image.size, Image.Resampling.BICUBIC), {'operation': operation, 'scale': scale}
    if operation == 'tilt':
        angle = round(rng.uniform(-1.2, 1.2), 3)
        return image.rotate(angle, Image.Resampling.BICUBIC, expand=True, fillcolor='white'), {'operation': operation, 'angle': angle}
    return image.copy(), {'operation': 'clean'}


def prepare(annotations: Path, images: Path, output: Path, mode: str = 'smoke', seed: int = 42,
            augment: bool = False, max_train: int = 3000, max_val: int = 600) -> dict:
    if output.exists():
        raise FileExistsError(f'Use a new output directory: {output}')
    if max_train < 1 or max_val < 2:
        raise ValueError('Invalid subset limits')
    annotations, images = annotations.resolve(), images.resolve()
    source_counts: Counter = Counter()
    kept_counts: Counter = Counter()
    all_sources: dict[str, str] = {}
    rows = []
    ids = set()
    # Stream the full annotation file; retain only bounded subsets in memory.
    for row in iter_jsonl(annotations):
        group = document_id(row['filename'])
        source = row.get('split')
        if source not in {'train', 'val'}:
            raise ValueError(f'Unexpected source split: {source}')
        if group in all_sources and all_sources[group] != source:
            raise ValueError(f'Paper appears in different source splits: {group}')
        all_sources[group] = source
        source_counts[source] += 1
        limit = max_train if source == 'train' else max_val
        if kept_counts[source] >= limit:
            continue
        if row['filename'] in ids:
            raise ValueError(f'Duplicate sample: {row["filename"]}')
        ids.add(row['filename'])
        rows.append(row)
        kept_counts[source] += 1
    if mode == 'official' and not kept_counts['val']:
        raise ValueError('Official mode requires held-out val samples')
    mapping = allocate(rows, mode, seed)
    splits: dict[str, list[dict]] = {'train': [], 'dev': [], 'test': []}
    image_hash_splits: dict[str, str] = {}
    audited = []
    for raw in rows:
        sample_id = Path(raw['filename']).stem
        group = document_id(raw['filename'])
        split = mapping[group]
        # Accept both a flat image directory and the canonical split/image tree.
        image_path = images / raw['filename']
        if not image_path.is_file():
            image_path = images / raw['split'] / raw['filename']
        if not image_path.is_file() or not image_path.resolve().is_relative_to(images):
            raise ValueError(f'Missing or unsafe image: {image_path}')
        target = annotation_html(raw)
        image_hash = sha256_file(image_path)
        if image_hash in image_hash_splits and image_hash_splits[image_hash] != split:
            raise ValueError('Identical image crosses train/dev/test')
        image_hash_splits[image_hash] = split
        with Image.open(image_path) as source_image:
            image = source_image.convert('RGB')
            image.load()
            width, height = image.size
        audited.append((raw, sample_id, group, split, image_path, image_hash, width, height, target))
    if any(not any(row[3] == s for row in audited) for s in splits):
        raise ValueError('Each split must be nonempty; increase subset limits')
    output.mkdir(parents=True)
    for raw, sample_id, group, split, image_path, image_hash, width, height, target in audited:
        image_dest = output / 'images' / raw['filename']
        image_dest.parent.mkdir(exist_ok=True)
        transform = {'operation': 'clean'}
        if augment and split == 'train':
            with Image.open(image_path) as source_image:
                image, transform = augment_image(source_image.convert('RGB'), sample_id, seed)
                width, height = image.size
                if transform['operation'] != 'clean':
                    image.save(image_dest)
                else:
                    shutil.copyfile(image_path, image_dest)
        else:
            shutil.copyfile(image_path, image_dest)
        attrs = table_attributes(target)
        attrs['small_text_proxy'] = width / max(1, attrs['cells']/attrs['rows']) < 80
        record = {'id': sample_id, 'paper_id': group, 'image': 'images/' + raw['filename'],
                  'html': target, 'source_split': raw['split'], 'split': split,
                  'source_image_sha256': image_hash, 'image_sha256': sha256_file(image_dest),
                  'width': width, 'height': height, 'attributes': attrs, 'transform': transform}
        splits[split].append(record)
    for split, records in splits.items():
        records.sort(key=lambda r: _key(seed, r['id']))
        write_jsonl(output / f'{split}.jsonl', records)
        converted = [{'conversations': [{'from': 'human', 'value': '<image>\n' + PROMPT},
                                        {'from': 'gpt', 'value': r['html'].replace('<html><body>', '').replace('</body></html>', '')}],
                      'images': [r['image']]} for r in records]
        write_json(output / 'lf' / f'{split}.json', converted)
    write_json(output / 'lf/dataset_info.json', {
        f'table_{split}': {'file_name': f'{split}.json', 'formatting': 'sharegpt',
                          'columns': {'messages': 'conversations', 'images': 'images'}} for split in splits})
    receipt = {'mode': mode, 'benchmark': mode != 'smoke', 'seed': seed, 'augmentation': augment,
               'annotations_sha256': sha256_file(annotations), 'source_counts': dict(source_counts),
               'retained_source_counts': dict(kept_counts),
               'split_counts': {s: len(r) for s, r in splits.items()},
               'paper_ids': {s: sorted({r['paper_id'] for r in records}) for s, records in splits.items()},
               'manifest_sha256': {s: sha256_file(output/f'{s}.jsonl') for s in splits},
               'checks': {'paper_disjoint': True, 'cross_split_exact_image_disjoint': True},
               'limitations': ['No near-duplicate detector; model pretraining contamination is unknown.',
                               'Official mode uses bounded first-N train/val subsets, not a random full-data benchmark.',
                               'small_text_proxy is geometric, not an OCR-measured font size.']}
    if mode == 'smoke':
        receipt['limitations'].append('All official examples are source train; repartitioning is smoke only, never a final benchmark.')
    write_json(output / 'audit.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False))
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['smoke', 'official'], default='smoke')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--max-train', type=int, default=3000)
    parser.add_argument('--max-val', type=int, default=600)
    args = parser.parse_args()
    prepare(args.annotations, args.images, args.output, args.mode, args.seed, args.augment, args.max_train, args.max_val)


if __name__ == '__main__':
    main()
