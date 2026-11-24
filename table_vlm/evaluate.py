"""Evaluate every manifest row using the pinned upstream TEDS implementation."""

import argparse
import importlib.util
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from .html_utils import InvalidTable, canonical_table, cell_texts, NORMALIZATION_VERSION
from .io_utils import read_jsonl, sha256_file, unique_by_id, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[1]
TEDS_SOURCE = ROOT / 'research/upstream/opendatalab__OmniDocBench/src/metrics/table_metric.py'
TEDS_REVISION = 'f133a71e9e91c3621c7ce8994200a7b394a06eb3'


def load_teds() -> type:
    spec = importlib.util.spec_from_file_location('omnidocbench_pinned_table_metric', TEDS_SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load pinned upstream TEDS: {TEDS_SOURCE}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TEDS


def number_strings(text: str) -> list[str]:
    # Preserve the sign: PMC often uses U+2212 rather than ASCII hyphen-minus.
    normalized = text.translate(str.maketrans({'−': '-', '＋': '+', '－': '-'}))
    return re.findall(r'[+-]?(?:\d+(?:[.,]\d+)*|[.,]\d+)(?:%|‰)?', normalized)


def numeric_tokens(value: str) -> Counter:
    # Diagnostic only: index-aligned numeric tokens, not semantic field accuracy.
    tokens = Counter()
    for index, text in enumerate(cell_texts(value)):
        for token in number_strings(text):
            tokens[index, token] += 1
    return tokens


def evaluate(manifest: Path, predictions: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f'Use a new report directory: {output}')
    truth = unique_by_id(read_jsonl(manifest))
    pred = unique_by_id(read_jsonl(predictions))
    if not truth:
        raise ValueError('Empty evaluation manifest')
    if extra := set(pred) - set(truth):
        raise ValueError(f'Predictions contain IDs from a different split: {sorted(extra)[:3]}')
    TEDS = load_teds()
    metric, structure_metric = TEDS(), TEDS(structure_only=True)
    results = []
    numeric_tp = numeric_gt = numeric_pred = 0
    for sample_id, row in truth.items():
        # Bad annotations are a fatal data error, never silently dropped.
        target = canonical_table(row['html'])
        item = pred.get(sample_id)
        error = ''
        normalized = ''
        if item is None:
            error = 'missing_prediction'
        elif item.get('status', 'ok') != 'ok':
            error = str(item.get('status', 'generation_error'))
        else:
            try:
                normalized = canonical_table(item.get('html', ''))
            except (InvalidTable, ValueError, TypeError) as exc:
                error = 'invalid_html'
                error_detail = str(exc)
        full_score = structure_score = 0.0
        if normalized:
            full_score = float(metric.evaluate(normalized, target))
            structure_score = float(structure_metric.evaluate(normalized, target))
        gt_tokens = numeric_tokens(target)
        actual_tokens = numeric_tokens(normalized) if normalized else Counter()
        hits = sum((gt_tokens & actual_tokens).values())
        numeric_tp += hits
        numeric_gt += sum(gt_tokens.values())
        numeric_pred += sum(actual_tokens.values())
        if error:
            category = error
        elif structure_score < 1 - 1e-9:
            category = 'structure_mismatch'
        elif full_score < 1 - 1e-9:
            category = 'cell_content_mismatch'
        else:
            category = 'exact_teds_match'
        result = {'id': sample_id, 'teds': full_score, 'teds_s': structure_score,
                  'valid_html': bool(normalized), 'category': category,
                  'attributes': row.get('attributes', {}),
                  'numeric_tokens_gt': sum(gt_tokens.values()), 'numeric_tokens_matched': hits,
                  'prediction': item.get('html', '') if item else '', 'target': target}
        if error == 'invalid_html':
            result['error_detail'] = error_detail
        results.append(result)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in results:
        for key, value in row['attributes'].items():
            if isinstance(value, bool):
                grouped[f'{key}={value}'].append(row)
    total = len(truth)
    audit_path = manifest.parent / 'audit.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8')) if audit_path.exists() else {}
    if audit and audit.get('manifest_sha256', {}).get(manifest.stem) != sha256_file(manifest):
        raise ValueError('Evaluation manifest differs from its audit receipt')
    splits = {row.get('split') for row in truth.values()}
    heldout = audit.get('mode') == 'official' and len(splits) == 1 and splits <= {'dev', 'test'}
    report = {'samples': total, 'predictions_present': len(pred), 'missing_predictions': total-len(pred),
              'teds': sum(r['teds'] for r in results)/total,
              'teds_s': sum(r['teds_s'] for r in results)/total,
              'valid_html_rate': sum(r['valid_html'] for r in results)/total,
              'numeric_token_precision_by_cell_index': numeric_tp/numeric_pred if numeric_pred else None,
              'numeric_token_recall_by_cell_index': numeric_tp/numeric_gt if numeric_gt else None,
              'categories': dict(Counter(r['category'] for r in results)),
              'groups': {g: {'n': len(rs), 'teds': sum(r['teds'] for r in rs)/len(rs),
                             'teds_s': sum(r['teds_s'] for r in rs)/len(rs)} for g, rs in grouped.items()},
              'manifest_sha256': sha256_file(manifest), 'predictions_sha256': sha256_file(predictions),
              'normalization_version': NORMALIZATION_VERSION,
              'normalization_source_sha256': sha256_file(ROOT/'table_vlm/html_utils.py'),
              'upstream': {'repo': 'opendatalab/OmniDocBench', 'revision': TEDS_REVISION,
                           'file': 'src/metrics/table_metric.py', 'sha256': sha256_file(TEDS_SOURCE)},
              'dataset_mode': audit.get('mode', 'unspecified'),
              'quality_claim_allowed': heldout,
              'evaluation_split': sorted(str(s) for s in splits),
              'protocol': ['All manifest samples remain in the denominator; missing/invalid/error/truncated outputs score zero.',
                           'Complete balanced HTML is required; surrounding Markdown fence may be removed.',
                           'th is normalized to td on both sides for the upstream metric.',
                           'Nonstructural data-bbox metadata is ignored on both sides; no tag/row/cell/span repair.',
                           'Numeric diagnostics match tokens by flattened cell index; they are not semantic financial field accuracy.',
                           'Official mode is a local held-out subset, not an official leaderboard score.']}
    output.mkdir(parents=True)
    write_json(output/'summary.json', report)
    write_jsonl(output/'samples.jsonl', results)
    write_jsonl(output/'bad_cases.jsonl', sorted((r for r in results if r['category'] != 'exact_teds_match'),
                                               key=lambda r: (r['teds'], r['id'])))
    text = ('# 表格评测报告\n\n' +
            ('仅为流程冒烟验证，不能作为模型效果或简历指标。\n\n' if not report['quality_claim_allowed'] else '本地留出集评测，不是官方排行榜成绩。\n\n') +
            f"样本 {total}；缺失预测 {report['missing_predictions']}；TEDS {report['teds']:.4f}；"
            f"TEDS-S {report['teds_s']:.4f}；有效 HTML {report['valid_html_rate']:.1%}。\n\n" +
            '详细分组、数据指纹和协议见 summary.json；全部样本见 samples.jsonl；失败案例见 bad_cases.jsonl。\n')
    (output/'report.md').write_text(text, encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.manifest, args.predictions, args.output)


if __name__ == '__main__':
    main()
