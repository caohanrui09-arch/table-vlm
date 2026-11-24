"""Create an uploadable source/small-sample bundle, excluding environments and weights."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'dist/doc-table-vlm-start.zip')
    parser.add_argument('--cpu-stage', action='store_true', help='Use verified CPU-stage datasets under the existing smoke paths')
    parser.add_argument('--replace', action='store_true', help='Explicitly rebuild this generated output archive')
    parser.add_argument('--formal-root', type=Path, help='Add completed formal CPU data/results from the D-drive root')
    args = parser.parse_args()
    if args.formal_root:
        args.cpu_stage = True
    if args.output.exists() and not args.replace:
        raise FileExistsError(args.output)
    files = []
    aliases: dict[Path, Path] = {}
    for folder in ['table_vlm', 'tests', 'configs', 'research', 'scripts',
                   'data/raw/pubtabnet-examples']:
        for path in (ROOT/folder).rglob('*'):
            if not path.is_file() or '__pycache__' in path.parts or 'generated' in path.parts:
                continue
            files.append(path)
    for source, target in ([('data/prepared/cpu-clean-v2', 'data/prepared/smoke'),
                            ('data/prepared/cpu-aug-v2', 'data/prepared/smoke-aug')]
                           if args.cpu_stage else [('data/prepared/smoke', 'data/prepared/smoke'),
                                                   ('data/prepared/smoke-aug', 'data/prepared/smoke-aug')]):
        for path in (ROOT/source).rglob('*'):
            if path.is_file():
                files.append(path)
                aliases[path] = Path(target)/path.relative_to(ROOT/source)
    for name in ['README.md', 'AUTODL_STEPS.md', 'THIRD_PARTY_NOTICES.md',
                 'requirements-data.txt', 'requirements-data-lock.txt', 'requirements-train.txt']:
        files.append(ROOT/name)
    for name in ['CPU_PROGRESS.md', 'EXPERIMENT_PROTOCOL.md', 'DATA_POLICY.md', 'requirements-paddle-cpu.txt']:
        if (ROOT/name).is_file():
            files.append(ROOT/name)
    if args.cpu_stage:
        files.append(ROOT/'requirements-paddle-cpu-lock-windows.txt')
        for folder in ['reports/cpu-stage', 'runs/paddle-cpu-dev-v1']:
            files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file())
    if args.formal_root:
        formal = args.formal_root.resolve()
        if not (formal/'reports/paddle-dev/summary.json').is_file():
            raise FileNotFoundError('Formal dev evaluation must complete before making the final bundle')
        folders = [('prepared/formal-4032-clean-v1', 'data/prepared/formal-clean'),
                   ('prepared/formal-4032-aug-v1', 'data/prepared/formal-aug'),
                   ('reports', 'reports'), ('runs/paddle-dev', 'runs/paddle-dev'),
                   ('raw/archive-notices', 'research/full-dataset-notices')]
        for source, target in folders:
            for path in (formal/source).rglob('*'):
                if path.is_file() and path.name != 'delivery-verification.json':
                    files.append(path)
                    aliases[path] = Path(target)/path.relative_to(formal/source)
        for name in ['selected-annotations.jsonl', 'selected-annotations-native-4032.jsonl',
                     'candidate-selection.json', 'archive-inspection.json']:
            path = formal/'raw'/name
            files.append(path)
            aliases[path] = Path('data/raw/formal-selection')/name
        for name in ['FORMAL_CPU_PLAN.md', 'FORMAL_RUN_STEPS.md', 'INTERVIEW_GUIDE.md',
                     'requirements-data-formal-lock-windows.txt']:
            files.append(ROOT/name)
    if any(not p.is_file() for p in files):
        raise FileNotFoundError('A documented bundle input is missing')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, 'w' if args.replace else 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        receipt = []
        for path in sorted(files):
            relative = aliases[path] if path in aliases else path.relative_to(ROOT)
            payload = path.read_bytes()
            if path.suffix in {'.sh', '.py', '.md', '.txt', '.yaml', '.json', '.jsonl'} and (not args.formal_root or path.suffix == '.sh'):
                payload = payload.replace(b'\r\n', b'\n')
            if args.cpu_stage and path.name == 'CPU_PROGRESS.md':
                payload = payload.replace(b'data/prepared/cpu-clean-v2', b'data/prepared/smoke')
                payload = payload.replace(b'data/prepared/cpu-aug-v2', b'data/prepared/smoke-aug')
            name = 'doc-table-vlm/' + relative.as_posix()
            archive.writestr(name, payload)
            receipt.append({'path': name, 'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)})
        archive.writestr('doc-table-vlm/bundle_manifest.json', json.dumps(receipt, indent=2))
    print(json.dumps({'bundle': str(args.output), 'files': len(files), 'bytes': args.output.stat().st_size,
                      'includes_model_weights': False, 'includes_smoke_training_samples': True,
                      'includes_formal_cpu_data': bool(args.formal_root)}))


if __name__ == '__main__':
    main()
