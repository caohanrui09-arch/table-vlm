"""Acquire official safetensors without executing repository code; record hashes."""

import argparse
import json
import os
import subprocess
from pathlib import Path

from .io_utils import sha256_file, write_json


def download_from_manifest(output: Path, manifest: Path, transport: str, revision: str | None = None) -> dict:
    """Use previously verified metadata; contact only the selected weight host."""
    expected = json.loads(manifest.read_text(encoding='utf-8'))
    if expected['repo'] != 'Qwen/Qwen3-VL-2B-Instruct' or transport not in {'modelscope', 'curl'}:
        raise ValueError('Local manifest mode requires Qwen2B and modelscope/curl transport')
    if revision and revision != expected['revision']:
        raise ValueError('Requested revision differs from the pinned manifest')
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    seen = set()
    # Reject bad/missing processor metadata before spending time on weights.
    for item in sorted(expected['files'], key=lambda r: r['path'].endswith('.safetensors')):
        path = (output/item['path']).resolve()
        if not path.is_relative_to(output) or item['path'] in seen:
            raise ValueError('Unsafe or duplicate manifest path')
        seen.add(item['path'])
        if path.exists():
            if path.stat().st_size != item['bytes'] or sha256_file(path) != item['sha256']:
                raise ValueError(f'Existing model file differs: {item["path"]}')
            continue
        if not path.name.endswith('.safetensors'):
            raise FileNotFoundError(f'Upload the processor bootstrap first: {path.name}')
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(path.suffix+'.part')
        url = (f'https://modelscope.cn/models/{expected["repo"]}/resolve/master/{item["path"]}'
               if transport == 'modelscope' else
               f'https://huggingface.co/{expected["repo"]}/resolve/{expected["revision"]}/{item["path"]}')
        subprocess.run(['curl.exe' if os.name == 'nt' else 'curl', '--fail', '--location', '--progress-bar',
                        '--retry', '3', '--connect-timeout', '20', '--speed-limit', '1024', '--speed-time', '45',
                        '--max-time', '900', '--continue-at', '-', '--output', str(partial), url], check=True)
        if partial.stat().st_size != item['bytes'] or sha256_file(partial) != item['sha256']:
            raise ValueError(f'Downloaded weight differs from fixed manifest: {item["path"]}')
        partial.rename(path)
    receipt = output/'source_receipt.json'
    if receipt.exists():
        if json.loads(receipt.read_text(encoding='utf-8')) != expected:
            raise ValueError('Do not overwrite a different model receipt')
    else:
        write_json(receipt, expected)
    return {'repo': expected['repo'], 'revision': expected['revision'], 'files': len(expected['files']),
            'hashes_verified': True, 'metadata_network_requests': 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('models/Qwen3-VL-2B-Instruct'))
    parser.add_argument('--revision', help='Resolve current official revision once when omitted, then persist it')
    parser.add_argument('--metadata-manifest', type=Path, help='Previously verified metadata; avoids Hub API requests')
    parser.add_argument('--transport', choices=['hub', 'curl', 'modelscope'], default='hub',
                        help='Use resumable curl for large weights if the Hub HTTP transport stalls')
    args = parser.parse_args()
    if args.metadata_manifest:
        print(json.dumps(download_from_manifest(args.output, args.metadata_manifest, args.transport, args.revision)), flush=True)
        return
    from huggingface_hub import HfApi, snapshot_download
    repo = 'Qwen/Qwen3-VL-2B-Instruct'
    receipt_path = args.output / 'source_receipt.json'
    existing = json.loads(receipt_path.read_text(encoding='utf-8')) if receipt_path.exists() else None
    revision = args.revision or (existing['revision'] if existing else None)
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    sha = info.sha
    if existing and existing['revision'] != sha:
        raise ValueError('Do not change an existing model revision in place')
    patterns = ['*.json', '*.txt', 'LICENSE', 'README.md']
    if args.transport == 'hub':
        patterns.append('*.safetensors')
    snapshot_download(repo, revision=sha, local_dir=str(args.output), max_workers=2, allow_patterns=patterns)
    if args.transport in {'curl', 'modelscope'}:
        for item in info.siblings:
            if not item.rfilename.endswith('.safetensors'):
                continue
            path = args.output/item.rfilename
            path.parent.mkdir(parents=True, exist_ok=True)
            if item.lfs and path.exists() and path.stat().st_size == item.size:
                if sha256_file(path) == item.lfs.sha256:
                    continue
                raise ValueError(f'Existing weight hash mismatch: {path}')
            url = (f'https://modelscope.cn/models/{repo}/resolve/master/{item.rfilename}'
                   if args.transport == 'modelscope' else f'https://huggingface.co/{repo}/resolve/{sha}/{item.rfilename}')
            subprocess.run(['curl.exe' if os.name == 'nt' else 'curl', '--fail', '--location', '--silent', '--show-error',
                            '--retry', '3', '--connect-timeout', '20', '--speed-limit', '1024', '--speed-time', '45',
                            '--max-time', '900', '--continue-at', '-', '--output', str(path),
                            url], check=True)
    files = []
    for item in info.siblings:
        path = args.output / item.rfilename
        if not path.is_file():
            continue
        digest = sha256_file(path)
        lfs = item.lfs
        expected = (getattr(lfs, 'sha256', None) if lfs else None)
        if expected and digest != expected:
            raise ValueError(f'Weight hash mismatch: {item.rfilename}')
        files.append({'path': item.rfilename, 'sha256': digest, 'bytes': path.stat().st_size,
                      'lfs_sha256_verified': bool(expected)})
    receipt = {'repo': repo, 'revision': sha, 'files': files,
               'source': f'https://huggingface.co/{repo}/tree/{sha}'}
    if not receipt_path.exists():
        write_json(receipt_path, receipt)
    elif existing != receipt:
        raise ValueError('Model receipt changed; investigate files rather than overwriting evidence')
    print(json.dumps({'repo': repo, 'revision': sha, 'files': len(files), 'hashes_verified': True}))


if __name__ == '__main__':
    main()
