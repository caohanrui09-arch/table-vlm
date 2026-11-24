"""Fetch three small official Paddle inference models with bounded safe extraction."""

import argparse
import hashlib
import io
import json
import tarfile
import urllib.request
from pathlib import Path, PurePosixPath

from .io_utils import sha256_file, write_json


MODELS = ['SLANet_plus', 'PP-OCRv4_mobile_det', 'PP-OCRv4_mobile_rec']
SOURCE = 'https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('models/paddle-cpu'))
    args = parser.parse_args()
    for name in MODELS:
        root = (args.output/name).resolve()
        receipt_path = root/'source_receipt.json'
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
            if all(sha256_file(root/r['path']) == r['sha256'] for r in receipt['files']):
                print(json.dumps({'model': name, 'cached_hashes_verified': True}), flush=True)
                continue
            raise ValueError(f'Cached model content changed: {name}')
        url = SOURCE+name+'_infer.tar'
        print(f'Downloading official CPU model: {name}', flush=True)
        with urllib.request.urlopen(url, timeout=120) as response:
            payload = response.read(64*1024*1024+1)
        if len(payload) > 64*1024*1024:
            raise ValueError('Model archive exceeds expected size bound')
        root.mkdir(parents=True, exist_ok=True)
        files = []
        with tarfile.open(fileobj=io.BytesIO(payload), mode='r:*') as archive:
            members = archive.getmembers()
            if len(members) > 50 or sum(m.size for m in members) > 128*1024*1024:
                raise ValueError('Unexpected archive size/member count')
            for member in members:
                parts = PurePosixPath(member.name).parts
                if not parts or '..' in parts or PurePosixPath(member.name).is_absolute() or '\\' in member.name:
                    raise ValueError('Unsafe archive path')
                if member.isdir():
                    continue
                if not member.isfile() or len(parts) < 2:
                    raise ValueError('Unexpected archive member')
                relative = Path(*parts[1:])
                target = (root/relative).resolve()
                if not target.is_relative_to(root):
                    raise ValueError('Archive member escapes model directory')
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError('Unreadable archive member')
                content = source.read()
                if target.exists():
                    if target.read_bytes() != content:
                        raise ValueError('Refuse to overwrite different model file')
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open('xb') as stream:
                        stream.write(content)
                files.append({'path': relative.as_posix(), 'sha256': sha256_file(target), 'bytes': len(content)})
        write_json(receipt_path, {'model': name, 'source_url': url,
                   'archive_sha256': hashlib.sha256(payload).hexdigest(), 'files': files,
                   'integrity_note': 'Recorded SHA256 for reproducibility; publisher checksum not supplied by this URL.'})
        print(json.dumps({'model': name, 'files': len(files), 'archive_bytes': len(payload)}), flush=True)


if __name__ == '__main__':
    main()
