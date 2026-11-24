"""Run the existing PaddleX PP-OCR+SLANet pipeline on CPU and retain every result."""

import argparse
import importlib.metadata
import json
import os
import time
from pathlib import Path

from .io_utils import read_jsonl, sha256_file, unique_by_id, write_json


def authorize_split(rows: list[dict], manifest: Path, config: Path, frozen_test: Path | None) -> bool:
    if not rows:
        raise ValueError('Empty dataset')
    splits={r['split'] for r in rows}
    if 'test' not in splits:
        if frozen_test is not None:
            raise ValueError('Final-test authorization supplied for non-test data')
        return False
    if splits!={'test'} or frozen_test is None:
        raise ValueError('Test inference requires the explicit frozen final-test protocol')
    freeze=json.loads(frozen_test.read_text(encoding='utf-8'))
    if (freeze['selected_model']!='q1' or len(rows)!=freeze['test_samples']
        or sha256_file(manifest)!=freeze['test_manifest_sha256']
        or sha256_file(config)!=freeze['paddle']['config_sha256']):
        raise ValueError('Final-test data/configuration differs from the freeze')
    return True


def main() -> None:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['FLAGS_use_cuda'] = 'false'
    os.environ.setdefault('PADDLE_PDX_CACHE_HOME', str(Path('.cache/paddlex-cpu').resolve()))
    os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
    os.environ['PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT'] = 'False'
    from paddlex import create_pipeline
    from paddlex.inference.utils.pp_option import PaddlePredictorOption
    import paddle

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('configs/paddle_cpu.yaml'))
    parser.add_argument('--resume', action='store_true', help='Append only unprocessed IDs with identical data/config')
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--frozen-test', type=Path, help='Final-test freeze registered after dev-only model selection')
    args = parser.parse_args()
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    if paddle.is_compiled_with_cuda():
        raise RuntimeError('Use the independent paddlepaddle CPU wheel for this baseline')
    paddle.set_device('cpu')
    rows = read_jsonl(args.manifest)
    unique_by_id(rows)
    final_test = authorize_split(rows, args.manifest, args.config, args.frozen_test)
    if not 1 <= args.shards <= 8 or not 0 <= args.shard_index < args.shards:
        raise ValueError('Invalid CPU shard configuration')
    total_samples = len(rows)
    rows = rows[args.shard_index::args.shards]
    models = {p.parent.name: json.loads(p.read_text(encoding='utf-8')) for p in Path('models/paddle-cpu').glob('*/source_receipt.json')}
    for name, receipt in models.items():
        for item in receipt['files']:
            if sha256_file(Path('models/paddle-cpu')/name/item['path']) != item['sha256']:
                raise ValueError(f'Pretrained CPU model changed: {name}/{item["path"]}')
    binding = {'manifest_sha256': sha256_file(args.manifest), 'config_sha256': sha256_file(args.config),
               'samples': len(rows), 'device': 'cpu', 'cpu_threads': 4,
               'shards': args.shards, 'shard_index': args.shard_index, 'total_samples': total_samples,
               'model_receipts': models}
    if final_test:
        freeze=json.loads(args.frozen_test.read_text(encoding='utf-8'))
        if models!=freeze['paddle']['model_receipts']:
            raise ValueError('Frozen Paddle model receipts changed')
        binding['final_test_freeze_sha256']=sha256_file(args.frozen_test)
    binding_path = args.output.with_suffix('.binding.json')
    done = set()
    if args.output.exists():
        if not binding_path.exists() or json.loads(binding_path.read_text(encoding='utf-8')) != binding:
            raise ValueError('Resume data/config binding differs')
        previous = unique_by_id(read_jsonl(args.output))
        if set(previous)-{r['id'] for r in rows}:
            raise ValueError('Resume predictions contain unknown sample IDs')
        done = set(previous)
        completed_receipt = args.output.with_suffix('.run.json')
        if completed_receipt.exists():
            if done != {r['id'] for r in rows}:
                raise ValueError('Completed run receipt exists but predictions are incomplete')
            print(json.dumps({'already_complete': True, 'samples': len(done), 'new_predictions': 0}), flush=True)
            return
    else:
        write_json(binding_path, binding)
    option = PaddlePredictorOption()
    option.cpu_threads = 4
    option.run_mode = 'paddle'
    started = time.perf_counter()
    pipeline = create_pipeline(pipeline=str(args.config), device='cpu', pp_option=option, use_hpip=False)
    setup_seconds = time.perf_counter()-started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output.parent/'raw'
    raw_dir.mkdir(exist_ok=args.resume)
    with args.output.open('a' if args.resume else 'x', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            if row['id'] in done:
                continue
            image_path = (args.manifest.parent/row['image']).resolve()
            if not image_path.is_relative_to(args.manifest.parent.resolve()) or sha256_file(image_path) != row['image_sha256']:
                raise ValueError(f'Image identity failed: {row["id"]}')
            started = time.perf_counter()
            status, prediction, detail = 'ok', '', ''
            try:
                outputs = list(pipeline.predict(str(image_path)))
                if len(outputs) != 1 or len(outputs[0]['table_res_list']) != 1:
                    status, detail = 'table_count_error', 'Expected exactly one table for a cropped input'
                else:
                    prediction = outputs[0]['table_res_list'][0]['pred_html']
                    outputs[0].save_to_json(str(raw_dir/f'{row["id"]}.json'))
            except (OSError, ValueError, RuntimeError, KeyError) as exc:
                status, detail = 'generation_error', f'{type(exc).__name__}: {exc}'
            result = {'id': row['id'], 'html': prediction, 'status': status, 'detail': detail,
                      'seconds': time.perf_counter()-started, 'device': 'cpu'}
            stream.write(json.dumps(result, ensure_ascii=False)+'\n')
            stream.flush()
            print(json.dumps({k: v for k, v in result.items() if k != 'html'}, ensure_ascii=False), flush=True)
    write_json(args.output.with_suffix('.run.json'), {'method': 'PaddleX traditional table_recognition: PP-OCRv4-mobile + SLANet_plus',
        'device': paddle.get_device(), 'compiled_with_cuda': paddle.is_compiled_with_cuda(),
        'versions': {p: importlib.metadata.version(p) for p in ['paddlepaddle', 'paddleocr', 'paddlex']},
        'manifest_sha256': sha256_file(args.manifest), 'config_sha256': sha256_file(args.config),
        'setup_seconds': setup_seconds, 'cpu_threads': 4, 'mkldnn': False, 'samples': len(rows), 'models': models,
        'shards': args.shards, 'shard_index': args.shard_index, 'total_samples': total_samples,
        'notes': ['Cropped table regions supplied to both methods; no layout detection.',
                  'Existing pretrained models; no training performed.', 'Local held-out results only, no leaderboard claim.',
                  'CPU timings are not comparable to GPU timings as a speed ranking.']})


if __name__ == '__main__':
    main()
