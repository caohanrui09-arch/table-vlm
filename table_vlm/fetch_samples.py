"""Download the fixed, tiny official sample set and verify Git blob hashes."""

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from .io_utils import sha256_file, write_json


ROOT = Path(__file__).resolve().parents[1]


def fetch(output: Path) -> None:
    source = json.loads((ROOT / 'research/pubtabnet_examples_source.json').read_text(encoding='utf-8'))
    output = output.resolve()
    receipt = []
    for item in source['files']:
        path = (output / item['path']).resolve()
        if not path.is_relative_to(output):
            raise ValueError('Source path escapes output directory')
        if path.exists():
            data = path.read_bytes()
        else:
            last_error = None
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(item['url'], timeout=30) as response:
                        data = response.read(item['size'] + 1)
                    blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
                    if len(data) != item['size'] or blob != item['git_blob_sha1']:
                        raise ValueError(f'Incomplete/mismatched response: {item["path"]}, bytes={len(data)}, blob={blob}')
                    break
                except (urllib.error.URLError, TimeoutError, ValueError) as exc:
                    last_error = exc
                    time.sleep(attempt + 1)
            else:
                raise RuntimeError(f'Download failed: {item["url"]}: {last_error}') from last_error
        blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        if len(data) != item['size'] or blob != item['git_blob_sha1']:
            raise ValueError(f'Source integrity mismatch: {item["path"]}')
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        receipt.append({'path': item['path'], 'sha256': sha256_file(path), 'bytes': len(data)})
    metadata = {'source': source, 'files': receipt, 'benchmark': False}
    receipt_path = output / 'receipt.json'
    if receipt_path.exists():
        if json.loads(receipt_path.read_text(encoding='utf-8')) != metadata:
            raise ValueError('Existing receipt differs; use a new output directory')
    else:
        write_json(receipt_path, metadata)
    print(json.dumps({'verified_files': len(receipt), 'output': str(output), 'benchmark': False}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'data/raw/pubtabnet-examples')
    args = parser.parse_args()
    fetch(args.output)


if __name__ == '__main__':
    main()
