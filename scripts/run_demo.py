"""Cross-platform demo launcher that does not require PowerShell 7.

Run it with the UI environment's Python:

    python scripts/run_demo.py --gpu-python <cuda-env>/python.exe

All arguments are forwarded to ``table_vlm.demo``.  This is the portable
alternative to ``start_demo.ps1`` (Windows PowerShell 7) and
``scripts/run_demo_autodl.sh`` (bash).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--python', type=Path, help='UI interpreter; defaults to the current Python')
    parser.add_argument('--image', help='run one image instead of starting the UI')
    parser.add_argument('--model', help='base model directory')
    parser.add_argument('--adapter', help='Q1 adapter directory')
    parser.add_argument('--gpu-python', help='model worker interpreter (may differ from the UI one)')
    parser.add_argument('--extra-pythonpath', help='worker-only package directories, os.pathsep-separated')
    parser.add_argument('--output-root', help='request/output directory')
    parser.add_argument('--example', help='override the bundled example image')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=7861)
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    parser.add_argument('--timeout', type=float, default=900)
    parser.add_argument('--skip-preflight', action='store_true')
    args = parser.parse_args()

    python = Path(args.python).resolve() if args.python else Path(sys.executable)
    command = [str(python), '-m', 'table_vlm.demo',
               '--host', args.host, '--port', str(args.port),
               '--device', args.device, '--timeout', str(args.timeout)]
    for flag, value in (
        ('--image', args.image), ('--model', args.model), ('--adapter', args.adapter),
        ('--gpu-python', args.gpu_python), ('--extra-pythonpath', args.extra_pythonpath),
        ('--output-root', args.output_root), ('--example', args.example),
    ):
        if value:
            command += [flag, str(value)]
    if args.skip_preflight:
        command.append('--skip-preflight')
    env = {**os.environ, 'PYTHONPATH': str(ROOT)}
    print(' '.join(command), flush=True)
    raise SystemExit(subprocess.call(command, cwd=ROOT, env=env))


if __name__ == '__main__':
    main()
