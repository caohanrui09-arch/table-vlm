"""Execute the authorized D-drive CPU preparation; no GPU or paid services."""

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from .io_utils import sha256_file, write_json


PROJECT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--wait-seconds', type=int, default=3600)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError('Use 1..4 CPU workers, four threads each')
    root = args.root.resolve()
    if os.name == 'nt' and root.drive.upper() != 'D:':
        raise ValueError('User requires the formal data/output directory on D drive')
    logs = root/'logs'
    logs.mkdir(parents=True, exist_ok=True)
    state_path = root/'pipeline-state.json'
    if state_path.exists():
        raise FileExistsError('A pipeline state already exists; inspect it before restarting individual stages')
    state = {'started_utc': datetime.now(timezone.utc).isoformat(), 'gpu_used': False, 'cloud_spending': 0,
             'root': str(root), 'workers': args.workers, 'cpu_threads_per_paddle_worker': 4,
             'source_sha256': {p.name: sha256_file(p) for p in (PROJECT/'table_vlm').glob('*.py')}}
    children: list[subprocess.Popen] = []
    handles = []
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='-1', HF_HUB_OFFLINE='1', PYTHONUNBUFFERED='1',
               PYTHONPATH=str(PROJECT/'.inference-deps')+os.pathsep+str(PROJECT),
               HF_HOME=str(root/'cache/huggingface'), PADDLE_PDX_CACHE_HOME=str(root/'cache/paddlex'),
               TEMP=str(root/'cache/tmp'), TMP=str(root/'cache/tmp'))
    (root/'cache/tmp').mkdir(parents=True, exist_ok=True)
    data_python = PROJECT/'.venv/Scripts/python.exe'
    model_python = PROJECT.parent/'.venv/Scripts/python.exe'
    paddle_python = PROJECT/'.venv-paddle-cpu/Scripts/python.exe'

    def update(phase: str, **extra: object) -> None:
        state.update(phase=phase, updated_utc=datetime.now(timezone.utc).isoformat(), **extra)
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        print(json.dumps({'phase': phase, **extra}, ensure_ascii=False), flush=True)

    def step(name: str, command: list[str]) -> None:
        update(name)
        with (logs/f'{name}.log').open('x', encoding='utf-8') as log:
            result = subprocess.run(command, cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            raise RuntimeError(f'{name} exited {result.returncode}; see {logs/name}.log')

    try:
        update('waiting_verified_download')
        deadline = time.monotonic()+args.wait_seconds
        while not args.archive.exists() or args.archive.with_suffix(args.archive.suffix+'.aria2').exists():
            if time.monotonic() > deadline:
                raise TimeoutError('Archive download is not complete; partial files preserved')
            time.sleep(5)
        step('formal_data', [str(data_python), '-m', 'table_vlm.formal_data', '--archive', str(args.archive), '--root', str(root)])
        data = root/'prepared/formal-clean-v1'
        manifest = data/'dev.jsonl'
        commands = [('length_audit', [str(model_python), '-m', 'table_vlm.length_audit', '--model',
                     str(PROJECT/'models/Qwen3-VL-2B-Instruct'), '--data', str(data), '--output-dir', str(root/'reports/lengths')])]
        for index in range(args.workers):
            commands.append((f'paddle_{index}', [str(paddle_python), '-m', 'table_vlm.predict_paddle_cpu',
                '--manifest', str(manifest), '--output', str(root/f'runs/paddle-dev/shard-{index}/predictions.jsonl'),
                '--shards', str(args.workers), '--shard-index', str(index)]))
        update('length_and_paddle_cpu')
        for name, command in commands:
            log = (logs/f'{name}.log').open('x', encoding='utf-8')
            handles.append(log)
            child = subprocess.Popen(command, cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT)
            children.append(child)
        update('length_and_paddle_cpu', process_ids={name: process.pid for (name, _), process in zip(commands, children, strict=True)})
        last_print = 0.
        while any(p.poll() is None for p in children):
            if time.monotonic()-last_print > 30:
                complete = sum(sum(1 for _ in p.open(encoding='utf-8')) for p in (root/'runs/paddle-dev').glob('shard-*/predictions.jsonl'))
                update('length_and_paddle_cpu', paddle_predictions_written=complete)
                last_print = time.monotonic()
            time.sleep(2)
        for (name, _), process in zip(commands, children, strict=True):
            if process.returncode:
                raise RuntimeError(f'{name} exited {process.returncode}; inspect its D-drive log')
        shards = [str(root/f'runs/paddle-dev/shard-{i}/predictions.jsonl') for i in range(args.workers)]
        predictions = root/'runs/paddle-dev/predictions.jsonl'
        step('merge_predictions', [str(data_python), '-m', 'table_vlm.merge_paddle', '--manifest', str(manifest),
                                  '--inputs', *shards, '--output', str(predictions)])
        step('evaluate_dev', [str(data_python), '-m', 'table_vlm.evaluate', '--manifest', str(manifest),
                             '--predictions', str(predictions), '--output', str(root/'reports/paddle-dev')])
        step('error_analysis', [str(data_python), '-m', 'table_vlm.error_analysis', '--scores', str(root/'reports/paddle-dev/samples.jsonl'),
            '--predictions', str(predictions), '--output', str(root/'reports/errors')])
        step('review_page', [str(data_python), '-m', 'table_vlm.review_report', '--data', str(data), '--split', 'dev',
            '--scored-samples', str(root/'reports/paddle-dev/samples.jsonl'), '--ids-file', str(root/'reports/errors/review-ids.json'),
            '--output', str(root/'reports/dev-review.html')])
        update('cpu_pipeline_complete', finished_utc=datetime.now(timezone.utc).isoformat(),
               remaining=['assistant visual review', 'final project report', 'GPU Qwen/LoRA and augmentation effect experiments'])
    except BaseException as exc:
        for process in children:
            if process.poll() is None:
                process.terminate()
        update('failed_or_interrupted', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        for handle in handles:
            handle.close()


if __name__ == '__main__':
    main()
