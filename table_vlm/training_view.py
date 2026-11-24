"""Derive equal-exposure clean/aug training views; never filter held-out rows."""

import argparse
import json
from pathlib import Path

from .audit_dataset import verify_dataset
from .data import prepare
from .io_utils import read_jsonl, sha256_file, unique_by_id, write_json, write_jsonl


def retained_ids(clean: list[dict], augmented: list[dict], maximum: int) -> tuple[set[str], list[dict]]:
    a, b = unique_by_id(clean), unique_by_id(augmented)
    if set(a) != set(b):
        raise ValueError('Clean/aug length records have different IDs')
    allowed, excluded = set(), []
    for sample_id, row in a.items():
        if row['split'] != b[sample_id]['split']:
            raise ValueError('Clean/aug length split differs')
        if row['split'] != 'train':
            continue
        other = b[sample_id]
        if max(row['multimodal_tokens'], other['multimodal_tokens']) <= maximum:
            allowed.add(sample_id)
        else:
            excluded.append({'id': sample_id, 'clean_tokens': row['multimodal_tokens'],
                             'aug_tokens': other['multimodal_tokens'], 'reason': 'training_native_length_guard'})
    return allowed, excluded


def build(root: Path, cutoff: int = 4096, margin: int = 64) -> dict:
    if not 0 <= margin < cutoff:
        raise ValueError('Invalid token safety margin')
    guard = cutoff-margin
    length_sets = []
    for label, report in [('formal-clean-v1', 'lengths'), ('formal-aug-v1', 'lengths-aug')]:
        data = root/'prepared'/label
        verify_dataset(data)
        summary = json.loads((root/'reports'/report/'summary.json').read_text(encoding='utf-8'))
        if summary['lengths_sha256'] != sha256_file(root/'reports'/report/'lengths.jsonl'):
            raise ValueError('Length checkpoint changed')
        if any(summary['binding']['manifest_sha256'][s] != sha256_file(data/f'{s}.jsonl') for s in ['train', 'dev', 'test']):
            raise ValueError('Lengths and source dataset differ')
        length_sets.append(read_jsonl(root/'reports'/report/'lengths.jsonl'))
    allowed, excluded = retained_ids(*length_sets, guard)
    original = root/'raw/selected-annotations.jsonl'
    original_receipt = json.loads((root/'prepared/formal-clean-v1/audit.json').read_text())
    if sha256_file(original) != original_receipt['annotations_sha256']:
        raise ValueError('Original selection changed')
    annotations = root/f'raw/selected-annotations-native-{guard}.jsonl'
    raw = read_jsonl(original)
    selected = [r for r in raw if r['split'] != 'train' or Path(r['filename']).stem in allowed]
    write_jsonl(annotations, selected)
    outputs = []
    for label, augment in [('clean', False), ('aug', True)]:
        output = root/f'prepared/formal-{guard}-{label}-v1'
        receipt = prepare(annotations, root/'raw/candidate-images', output, 'official', 42, augment, 100000, 100000)
        receipt.update(native_token_guard=guard, planned_cutoff_len=cutoff, template_safety_margin=margin,
            source_prepared=f'../formal-{label}-v1',
            near_duplicate_check='Inherited cross-split ImageHash check from the original 3300-table selection; subset only',
            limitations=['Curated local subset, not official leaderboard.',
                         'Native processor length guard does not replace checking final LlamaFactory encoding.'])
        (output/'audit.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        for split in ['dev', 'test']:
            if sha256_file(output/f'{split}.jsonl') != sha256_file(root/f'prepared/formal-clean-v1/{split}.jsonl'):
                raise ValueError('Training length filtering altered held-out rows')
        audit = verify_dataset(output, root/f'prepared/formal-{guard}-clean-v1' if augment else None)
        write_json(root/f'reports/data/formal-{guard}-{label}-audit.json', audit)
        outputs.append(str(output))
    result = {'cutoff_len': cutoff, 'native_token_guard': guard, 'template_safety_margin': margin,
        'original_train': sum(r['split'] == 'train' for r in raw), 'retained_train': len(allowed),
        'excluded_train': excluded, 'heldout_unchanged': True, 'dev': original_receipt['split_counts']['dev'],
        'test': original_receipt['split_counts']['test'], 'outputs': outputs,
        'note': 'Train-only exclusion uses the union of clean/aug length failures, before GPU training. No truncation; original dataset preserved.'}
    write_json(root/'reports/data/training-length-policy.json', result)
    print(json.dumps(result, ensure_ascii=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--cutoff', type=int, default=4096)
    parser.add_argument('--margin', type=int, default=64)
    args = parser.parse_args()
    build(args.root.resolve(), args.cutoff, args.margin)


if __name__ == '__main__':
    main()
