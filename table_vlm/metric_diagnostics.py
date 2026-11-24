"""Exercise TEDS on controlled label mutations, never pretend these are predictions."""

import argparse
import json
import re
from pathlib import Path

from lxml import etree, html

from .audit_dataset import verify_dataset
from .evaluate import load_teds
from .html_utils import InvalidTable, canonical_table
from .io_utils import read_jsonl, write_json


def mutate(value: str, operation: str) -> str:
    if operation == 'same':
        return value
    if operation == 'empty':
        return ''
    if operation == 'truncated':
        return value[:len(value)//2]
    doc = html.document_fromstring(value)
    if operation == 'cell_text':
        cell = doc.xpath('//table//td')[0]
        cell.text = (cell.text or '')+'X'
    elif operation == 'span':
        cell = doc.xpath('//table//td')[0]
        cell.set('colspan', str(int(cell.get('colspan', '1'))+1))
    elif operation == 'missing_row':
        row = doc.xpath('//table//tr')[-1]
        row.getparent().remove(row)
    else:
        raise ValueError(operation)
    return etree.tostring(doc, encoding='unicode', method='html')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    verify_dataset(args.data)
    rows = sorted(read_jsonl(args.data/'train.jsonl'), key=lambda r: (r['attributes']['cells'], r['id']))[:3]
    TEDS = load_teds()
    metric, structural = TEDS(), TEDS(structure_only=True)
    cases = []
    for row in rows:
        truth = canonical_table(row['html'])
        for operation in ['same', 'cell_text', 'span', 'missing_row', 'empty', 'truncated']:
            value = mutate(truth, operation)
            try:
                normalized = canonical_table(value)
            except InvalidTable:
                normalized = ''
            score = metric.evaluate(normalized, truth) if normalized else 0
            score_s = structural.evaluate(normalized, truth) if normalized else 0
            if operation == 'same':
                passed = score == score_s == 1
            elif operation == 'cell_text':
                passed = score < 1 and score_s == 1
            elif operation in {'span', 'missing_row'}:
                passed = score_s < 1
            else:
                passed = score == score_s == 0 and not normalized
            cases.append({'source_train_id': row['id'], 'operation': operation,
                          'teds': score, 'teds_s': score_s, 'check_passed': bool(passed)})
    report = {'kind': 'controlled_label_mutation_test_not_model_predictions', 'cases': cases,
              'all_checks_passed': all(c['check_passed'] for c in cases),
              'notes': ['Selected the three smallest training tables for fast deterministic CPU checks.',
                        'These numbers verify metric behavior; they are NOT recognition accuracy or model improvements.',
                        'No dev/test labels were used to construct mutations.']}
    write_json(args.output, report)
    print(json.dumps({'cases': len(cases), 'all_checks_passed': report['all_checks_passed'], 'model_predictions': False}))
    if not report['all_checks_passed']:
        raise SystemExit('Metric diagnostic expectation failed')


if __name__ == '__main__':
    main()
