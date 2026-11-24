"""Create an offline, network-free HTML review report for train or dev data."""

import argparse
import base64
import html
import io
import json
from pathlib import Path

from PIL import Image

from .audit_dataset import verify_dataset
from .html_utils import canonical_table
from .io_utils import read_jsonl, write_json


def build(data: Path, output: Path, split: str = 'train', scored_samples: Path | None = None,
          ids_file: Path | None = None) -> None:
    if split not in {'train', 'dev'}:
        raise ValueError('Review train/dev only; keep test blind')
    if output.exists():
        raise FileExistsError(output)
    audit = verify_dataset(data)
    scores = {}
    if scored_samples:
        scores = {r['id']: r for r in read_jsonl(scored_samples)}
    cards = []
    template = []
    requested = set(json.loads(ids_file.read_text(encoding='utf-8'))) if ids_file else None
    for row in read_jsonl(data/f'{split}.jsonl'):
        if requested is not None and row['id'] not in requested:
            continue
        with Image.open(data/row['image']) as source:
            image = source.convert('RGB')
            image.thumbnail((1100, 800))
            stream = io.BytesIO()
            image.save(stream, format='PNG')
        image_uri = 'data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode()
        label = html.escape(row['id'])
        attrs = row['attributes']
        meta = f"{attrs['rows']} 行 / {attrs['cells']} 单元格 / 合并单元格 {attrs['merged']} / {row['width']}×{row['height']}"
        target = canonical_table(row['html'])
        target_fragment = target.removeprefix('<html><body>').removesuffix('</body></html>')
        scored = scores.get(row['id'])
        prediction = ''
        if scored:
            caption = html.escape(f"{scored['category']} · TEDS {scored['teds']:.4f} · TEDS-S {scored['teds_s']:.4f}")
            rendered = ''
            if scored['valid_html']:
                rendered = canonical_table(scored['prediction']).removeprefix('<html><body>').removesuffix('</body></html>')
            prediction = f'<h3>实际预测：{caption}</h3><div class="table-wrap">{rendered}</div><details><summary>预测源码</summary><pre>{html.escape(scored["prediction"])}</pre></details>'
        cards.append(f'<article><h2>{label}</h2><p>{html.escape(meta)}</p>'
                     f'<div class="pair"><section><h3>原图（缩略展示）</h3><img alt="{label}" src="{image_uri}"></section>'
                     f'<section><h3>标注 HTML 渲染</h3><div class="table-wrap">{target_fragment}</div></section></div>'
                     f'<details><summary>标注源码</summary><pre>{html.escape(target)}</pre></details>{prediction}</article>')
        template.append({'id': row['id'], 'split': split, 'human_status': 'unreviewed',
                         'checks': {'text': None, 'row_column': None, 'span': None, 'number': None},
                         'notes': '', 'reviewer': ''})
    style = ('body{font-family:system-ui,sans-serif;margin:28px;color:#18222e;background:#f4f6f8}'
             'article{background:white;padding:24px;margin:22px 0;border:1px solid #d7dde4;border-radius:10px}'
             '.pair{display:grid;grid-template-columns:1fr 1fr;gap:22px}img{max-width:100%;height:auto}'
             '.table-wrap,pre{overflow:auto;max-height:700px}table{border-collapse:collapse;font-size:13px}'
             'td,th{border:1px solid #aeb8c4;padding:5px}pre{white-space:pre-wrap;font-size:12px}'
             '@media(max-width:900px){.pair{grid-template-columns:1fr}}')
    page = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
            '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; img-src data:; style-src \'unsafe-inline\'">'
            f'<title>表格数据检查 · {split}</title><style>{style}</style><body>'
            f'<h1>表格数据检查 · {split}</h1><p>当前数据模式：{html.escape(audit["mode"])}。'
            '仅检查数据与已有预测，不自动给出人工复核结论；测试集不展示。所有图片内嵌，无外部网络资源。</p>'
            + ''.join(cards) + '</body></html>')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding='utf-8')
    write_json(output.with_suffix('.review-template.json'), template)
    print(json.dumps({'report': str(output), 'split': split, 'samples': len(cards), 'human_review_completed': False}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--split', choices=['train', 'dev'], default='train')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scored-samples', type=Path)
    parser.add_argument('--ids-file', type=Path, help='JSON list of selected train/dev IDs')
    args = parser.parse_args()
    build(args.data, args.output, args.split, args.scored_samples, args.ids_file)


if __name__ == '__main__':
    main()
