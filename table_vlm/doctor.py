"""Preflight checks for the single-table demo.

``table-vlm-doctor`` answers one question before the UI or a GPU run starts:
*is this machine actually able to run the frozen Q1 demo?*  Every check is
reported instead of raising, so a missing 4.25GB weight does not hide a broken
adapter hash.

Examples
--------
    python -m table_vlm.doctor
    python -m table_vlm.doctor --deep --json
    python -m table_vlm.doctor --model D:/models/Qwen3-VL-2B-Instruct --gpu-python .venv/Scripts/python.exe
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

from . import paths
from .io_utils import sha256_file


OK = 'ok'
WARN = 'warn'
FAIL = 'fail'


def _check(name: str, status: str, detail: str, hint: str = '') -> dict:
    return {'name': name, 'status': status, 'detail': detail, 'hint': hint}


def check_python() -> dict:
    version = sys.version_info
    detail = f'{version.major}.{version.minor}.{version.micro} ({sys.executable})'
    if version < (3, 10):
        return _check('python', FAIL, detail, 'Use Python 3.10+; the handoff was verified on 3.12.')
    if version < (3, 12):
        return _check('python', WARN, detail, 'Python 3.12 is the verified version; older versions may work.')
    return _check('python', OK, detail)


def _read_receipt(model_dir: Path) -> dict | None:
    path = model_dir / 'source_receipt.json'
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None


def check_model(model_dir: Path, manifest: Path | None = None, deep: bool = False) -> dict:
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        return _check('model', FAIL, f'missing directory: {model_dir}',
                      'Download the fixed base model (see handoff/START_HERE.md).')
    missing = [name for name in paths.MODEL_METADATA if not (model_dir / name).is_file()]
    if missing:
        return _check('model', FAIL, 'missing metadata: ' + ', '.join(missing),
                      'Copy the pinned processor files or rerun the downloader.')
    weight = model_dir / paths.MODEL_WEIGHT
    if not weight.is_file():
        return _check('model', FAIL, f'missing weights: {weight.name}',
                      'Run: python -m table_vlm.download_model --metadata-manifest research/qwen2b-fixed-manifest.json --transport modelscope')
    size = weight.stat().st_size
    expected_bytes = None
    if manifest is not None and Path(manifest).is_file():
        try:
            rows = json.loads(Path(manifest).read_text(encoding='utf-8'))
            expected_bytes = next((row['bytes'] for row in rows['files']
                                   if row['path'] == paths.MODEL_WEIGHT), None)
        except (OSError, ValueError, KeyError):
            expected_bytes = None
    if expected_bytes is not None:
        if size != expected_bytes:
            return _check('model', FAIL, f'{weight.name} is {size} bytes, manifest says {expected_bytes}',
                          'A partial or corrupted download was found; remove it and download again.')
    elif size < 1024 ** 3:
        return _check('model', FAIL, f'{weight.name} is only {size} bytes',
                      'The base weight is about 4.25GB; the download is incomplete.')

    receipt = _read_receipt(model_dir)
    if receipt is None:
        return _check('model', WARN, f'{weight.name} present ({size} bytes), no source_receipt.json',
                      'The demo records the pinned revision; rerun the downloader to create the receipt.')
    revision = receipt.get('revision')
    if revision != paths.BASE_REVISION:
        return _check('model', FAIL, f'receipt revision is {revision!r}, expected {paths.BASE_REVISION}',
                      'Do not mix a different base revision with the frozen Q1 adapter.')
    if not deep:
        return _check('model', OK, f'{weight.name} {size} bytes, revision {revision[:12]}')
    for row in receipt.get('files', []):
        path = model_dir / row['path']
        if not path.is_file():
            return _check('model', FAIL, f'receipt file missing: {row["path"]}', 'Re-run the downloader.')
        if path.stat().st_size != row['bytes'] or sha256_file(path) != row['sha256']:
            return _check('model', FAIL, f'hash mismatch: {row["path"]}', 'Do not use a different model build.')
    return _check('model', OK, f'{len(receipt.get("files", []))} files deep-verified, revision {revision[:12]}')


def check_adapter(adapter_dir: Path, expected_sha: str = paths.Q1_SHA256) -> dict:
    adapter_dir = Path(adapter_dir)
    if not adapter_dir.is_dir():
        return _check('adapter', FAIL, f'missing directory: {adapter_dir}',
                      'The Q1 adapter ships in artifacts/q1-adapter.')
    missing = [name for name in paths.ADAPTER_FILES if not (adapter_dir / name).is_file()]
    if missing:
        return _check('adapter', FAIL, 'missing: ' + ', '.join(missing),
                      'Restore the adapter from the handoff package.')
    try:
        config = json.loads((adapter_dir / 'adapter_config.json').read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        return _check('adapter', FAIL, f'adapter_config.json is unreadable: {error}')
    digest = sha256_file(adapter_dir / 'adapter_model.safetensors')
    if digest != expected_sha:
        return _check('adapter', FAIL, f'hash {digest[:12]}... does not match frozen Q1 {expected_sha[:12]}...',
                      'Do not edit or retrain the adapter for the demo.')
    detail = (f'r{config.get("r")} alpha{config.get("lora_alpha")} '
              f'{"/".join(config.get("target_modules", []))}, sha {digest[:12]}...')
    return _check('adapter', OK, detail)


_RUNTIME_PROBE = r'''
import json, sys
info = {"python": sys.version.split()[0], "executable": sys.executable}
try:
    import torch
    info["torch"] = torch.__version__
    info["cuda"] = bool(torch.cuda.is_available())
    info["device"] = torch.cuda.get_device_name(0) if info["cuda"] else None
except Exception as error:
    info["torch_error"] = f"{type(error).__name__}: {error}"
try:
    import torchvision
    info["torchvision"] = torchvision.__version__
except Exception as error:
    info["torchvision_error"] = f"{type(error).__name__}: {error}"
for name in ("transformers", "peft", "accelerate"):
    try:
        info[name] = __import__("importlib.metadata", fromlist=["version"]).version(name)
    except Exception:
        info[name] = None
print(json.dumps(info))
'''


def check_runtime(python: Path, require_cuda: bool = False, timeout: int = 180,
                  extra_pythonpath: list[Path] | None = None) -> dict:
    python = Path(python)
    executable = str(python)
    if not python.is_file() and shutil.which(executable) is None:
        return _check('runtime', FAIL, f'interpreter not found: {python}',
                      'Pass --gpu-python or set TABLE_VLM_GPU_PYTHON.')
    env = dict(os.environ)
    if extra_pythonpath:
        existing = env.get('PYTHONPATH', '')
        env['PYTHONPATH'] = os.pathsep.join([*map(str, extra_pythonpath), *([existing] if existing else [])])
    try:
        completed = subprocess.run([executable, '-c', _RUNTIME_PROBE],
                                   capture_output=True, text=True, timeout=timeout, env=env,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    except (OSError, subprocess.TimeoutExpired) as error:
        return _check('runtime', FAIL, f'probe failed: {error}',
                      'Make sure this interpreter can import torch without activation scripts.')
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout).strip().splitlines()[-1:] or ['']
        return _check('runtime', FAIL, f'probe exited {completed.returncode}: {tail[0]}',
                      'Install the GPU extras: requirements-data.txt plus torch/torchvision/transformers/peft/accelerate.')
    try:
        info = json.loads(completed.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return _check('runtime', FAIL, 'probe returned unreadable output', 'Run the probe manually to debug.')
    if info.get('torch') is None:
        return _check('runtime', FAIL, 'torch is not importable in this interpreter',
                      'Install torch for the host (CUDA 12.4 wheels for the verified setup).')
    missing = [name for name in ('transformers', 'peft') if not info.get(name)]
    if missing:
        return _check('runtime', FAIL, 'missing packages: ' + ', '.join(missing),
                      'Install transformers==4.57.1 and peft==0.18.1 (+ accelerate).')
    if not info.get('torchvision'):
        return _check('runtime', FAIL, 'torchvision is not importable; the Qwen3-VL processor needs it',
                      'Install torchvision==0.21.0 for torch 2.6.0, or pass --extra-pythonpath to a directory that has it.')
    if not info.get('cuda'):
        status = FAIL if require_cuda else WARN
        return _check('runtime', status,
                      f'torch {info["torch"]}, torchvision {info["torchvision"]}, CPU only; no CUDA device',
                      'The frozen benchmark ran on CUDA. Use a GPU host or --device cpu for a slow smoke test.')
    return _check('runtime', OK,
                  f'torch {info["torch"]}, CUDA {info.get("device")}, '
                  f'torchvision {info["torchvision"]}, '
                  f'transformers {info.get("transformers")}, peft {info.get("peft")}')


def check_output(output: Path) -> dict:
    output = Path(output)
    try:
        output.mkdir(parents=True, exist_ok=True)
        probe = output / '.doctor-write-test'
        probe.write_text('ok', encoding='utf-8')
        probe.unlink()
    except OSError as error:
        return _check('output', FAIL, f'{output} is not writable: {error}',
                      'Choose a writable --output-root (avoid Program Files).')
    return _check('output', OK, str(output))


def check_port(host: str, port: int) -> dict:
    target_host = '127.0.0.1' if host in {'0.0.0.0', '::'} else host
    try:
        with socket.create_connection((target_host, port), timeout=1.5):
            return _check('port', WARN, f'{target_host}:{port} is already in use',
                          'Stop the other process or choose another --port.')
    except OSError:
        return _check('port', OK, f'{target_host}:{port} is free')


def run_doctor(model: Path | None = None, adapter: Path | None = None,
               gpu_python: Path | None = None, output: Path | None = None,
               manifest: Path | None = None,
               extra_pythonpath: str | Path | None = None,
               host: str = '127.0.0.1', port: int = 7861, deep: bool = False,
               require_cuda: bool = False, skip_runtime: bool = False,
               skip_port: bool = False) -> dict:
    model = paths.model_path(model)
    adapter = paths.adapter_path(adapter)
    gpu_python = paths.gpu_python(gpu_python)
    output = paths.output_path(output)
    manifest = paths.manifest_path(manifest)
    extras = paths.extra_pythonpath(extra_pythonpath)
    checks = [check_python(), check_model(model, manifest, deep=deep),
              check_adapter(adapter), check_output(output)]
    if not skip_runtime:
        checks.append(check_runtime(gpu_python, require_cuda=require_cuda, extra_pythonpath=extras))
    if not skip_port:
        checks.append(check_port(host, port))
    ok = all(item['status'] != FAIL for item in checks)
    return {'ok': ok, 'checks': checks,
            'paths': {'model': str(model), 'adapter': str(adapter),
                      'gpu_python': str(gpu_python), 'output': str(output),
                      'extra_pythonpath': [str(path) for path in extras]}}


def format_report(report: dict) -> str:
    marks = {OK: '[ok]  ', WARN: '[warn]', FAIL: '[fail]'}
    lines = []
    for item in report['checks']:
        lines.append(f'{marks[item["status"]]} {item["name"]}: {item["detail"]}')
        if item['hint'] and item['status'] != OK:
            lines.append(f'         hint: {item["hint"]}')
    if report['ok']:
        lines.append('Result: READY (warnings do not block the demo)')
    else:
        failed = sum(item['status'] == FAIL for item in report['checks'])
        lines.append(f'Result: NOT READY ({failed} failing check(s))')
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', type=Path, help=f'base model directory (env {paths.ENV_MODEL})')
    parser.add_argument('--adapter', type=Path, help=f'Q1 adapter directory (env {paths.ENV_ADAPTER})')
    parser.add_argument('--gpu-python', type=Path, help=f'model worker interpreter (env {paths.ENV_GPU_PYTHON})')
    parser.add_argument('--output-root', type=Path, help=f'demo output directory (env {paths.ENV_OUTPUT})')
    parser.add_argument('--manifest', type=Path, help='pinned model manifest used to check the weight size')
    parser.add_argument('--extra-pythonpath', help=f'worker-only package directories (os.pathsep-separated); '
                        f'also settable with {paths.ENV_EXTRA_PYTHONPATH}')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=7861)
    parser.add_argument('--deep', action='store_true', help='hash every model file (slow, 4.25GB)')
    parser.add_argument('--require-cuda', action='store_true', help='treat CPU-only runtime as a failure')
    parser.add_argument('--skip-runtime', action='store_true')
    parser.add_argument('--skip-port', action='store_true')
    parser.add_argument('--json', action='store_true', help='machine-readable output')
    args = parser.parse_args()
    report = run_doctor(model=args.model, adapter=args.adapter, gpu_python=args.gpu_python,
                        output=args.output_root, manifest=args.manifest,
                        extra_pythonpath=args.extra_pythonpath,
                        host=args.host, port=args.port,
                        deep=args.deep, require_cuda=args.require_cuda,
                        skip_runtime=args.skip_runtime, skip_port=args.skip_port)
    print(json.dumps(report, indent=2, ensure_ascii=False) if args.json else format_report(report))
    raise SystemExit(0 if report['ok'] else 1)


if __name__ == '__main__':
    main()
