"""Build a release zip together with a SHA256/size manifest.

Usage:
    python scripts/package_release.py
    python scripts/package_release.py --output dist/table-vlm-v0.2.0.zip

Virtual environments, caches, demo output and the 4.25GB base weight are
excluded.  The written zip is re-opened and every payload is hashed again, so
the manifest describes exactly what a recipient receives.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXCLUDE_DIRS = {'.git', '.cache', '__pycache__', '.venv', '.pytest_cache', '.ruff_cache',
                'demo-output', 'dist', 'release', 'node_modules'}
EXCLUDE_SUFFIXES = {'.pyc', '.pyo', '.part'}
EXCLUDE_RELATIVE = {Path('models/Qwen3-VL-2B-Instruct/model.safetensors')}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def iter_payload(project: Path):
    for path in sorted(project.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(project)
        if any(part in EXCLUDE_DIRS for part in relative.parts):
            continue
        if relative in EXCLUDE_RELATIVE or path.suffix in EXCLUDE_SUFFIXES:
            continue
        yield path, relative


def build(project: Path, output: Path, manifest_path: Path, prefix: str) -> dict:
    project = project.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path, relative in iter_payload(project):
            arcname = f'{prefix}/{relative.as_posix()}'
            archive.write(path, arcname)
            entries.append({'path': arcname, 'bytes': path.stat().st_size, 'sha256': sha256_file(path)})

    with zipfile.ZipFile(output) as archive:
        broken = archive.testzip()
        if broken:
            raise RuntimeError(f'CRC check failed for {broken}')
        for entry in entries:
            payload = archive.read(entry['path'])
            if len(payload) != entry['bytes'] or hashlib.sha256(payload).hexdigest() != entry['sha256']:
                raise RuntimeError(f'Payload changed after packaging: {entry["path"]}')

    manifest = {
        'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'project': prefix,
        'zip': output.name,
        'zip_sha256': sha256_file(output),
        'zip_bytes': output.stat().st_size,
        'total_files': len(entries),
        'total_bytes': sum(entry['bytes'] for entry in entries),
        'files': entries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project', type=Path, default=PROJECT_ROOT)
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT / 'dist/doc-table-vlm-improved-20261122.zip')
    parser.add_argument('--manifest', type=Path, help='defaults to <zip>.manifest.json')
    parser.add_argument('--prefix', default='table-vlm', help='top-level folder inside the zip')
    args = parser.parse_args()
    manifest_path = args.manifest or args.output.with_suffix('.manifest.json')
    manifest = build(args.project, args.output, manifest_path, args.prefix)
    print(json.dumps({'zip': str(args.output), 'manifest': str(manifest_path),
                      'zip_sha256': manifest['zip_sha256'], 'files': manifest['total_files'],
                      'bytes': manifest['zip_bytes']}, indent=2))


if __name__ == '__main__':
    main()
