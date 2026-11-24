"""Describe actual dev errors and select cases for visual review, without inventing human labels."""

import argparse
import difflib
import json
import statistics
from collections import Counter
from pathlib import Path

from .html_utils import canonical_table, cell_texts, table_attributes
from .evaluate import number_strings
from .io_utils import read_jsonl, sha256_file, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scores', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    samples = read_jsonl(args.scores)
    predictions = read_jsonl(args.predictions)
    details = []
    for row in samples:
        detail = {'id': row['id'], 'category': row['category'], 'teds': row['teds'], 'teds_s': row['teds_s'],
                  'assistant_visual_review': 'pending', 'human_review': 'unreviewed'}
        if row['valid_html']:
            target = cell_texts(row['target'])
            actual = cell_texts(canonical_table(row['prediction']))
            detail.update(gt_cells=len(target), pred_cells=len(actual),
                gt_structure=table_attributes(row['target']), pred_structure=table_attributes(canonical_table(row['prediction'])))
            if row['teds_s'] >= 1-1e-9:
                mismatches = []
                for index, (a, b) in enumerate(zip(target, actual, strict=True)):
                    if a != b:
                        nums_a, nums_b = number_strings(a), number_strings(b)
                        mismatches.append({'cell_index': index, 'target': a, 'prediction': b,
                                           'numeric_token_difference': nums_a != nums_b})
                detail['aligned_text_mismatches'] = mismatches
            else:
                detail['alignment_note'] = 'Different structures: flattened cell positions cannot establish field-level correspondence.'
                matcher = difflib.SequenceMatcher(a=target, b=actual, autojunk=False)
                detail['text_sequence_edit_blocks_diagnostic_only'] = [
                    {'operation': tag, 'gt_range': [a, b], 'pred_range': [c, d]}
                    for tag, a, b, c, d in matcher.get_opcodes() if tag != 'equal']
        details.append(detail)
    selected = []
    for category in ['structure_mismatch', 'cell_content_mismatch', 'invalid_html', 'generation_error']:
        selected.extend([r['id'] for r in sorted(details, key=lambda r: (r['teds'], r['id'])) if r['category'] == category][:5])
    ordered = sorted(details, key=lambda r: (r['teds'], r['id']))
    for rank in [.25, .5, .75, 1.]:
        if ordered:
            selected.append(ordered[round((len(ordered)-1)*rank)]['id'])
    selected = list(dict.fromkeys(selected))
    times = [r['seconds'] for r in predictions if isinstance(r.get('seconds'), (int, float))]
    args.output.mkdir(parents=True)
    write_jsonl(args.output/'error-details.jsonl', details)
    write_json(args.output/'review-ids.json', selected)
    summary = {'samples': len(samples), 'categories': dict(Counter(r['category'] for r in details)),
        'status_counts': dict(Counter(r.get('status', 'unspecified') for r in predictions)),
        'cpu_seconds': {'total': sum(times), 'median': statistics.median(times) if times else None,
                        'min': min(times) if times else None, 'max': max(times) if times else None},
        'numeric_mismatch_cells_with_identical_structure': sum(sum(x['numeric_token_difference'] for x in r.get('aligned_text_mismatches', [])) for r in details),
        'scores_sha256': sha256_file(args.scores), 'predictions_sha256': sha256_file(args.predictions),
        'selected_visual_review_cases': len(selected), 'human_review_completed': False,
        'interpretation': ['TEDS-S<1 implies a structural edit, not an automatic diagnosis of which merge is wrong.',
                           'TEDS-S=1 with TEDS<1 indicates content mismatch under the chosen normalization.',
                           'Numeric text mismatches are diagnostics, not business-field accuracy.',
                           'Blur/resolution/tilt may affect recognition; these runs do not establish causal enhancement gains.']}
    write_json(args.output/'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
