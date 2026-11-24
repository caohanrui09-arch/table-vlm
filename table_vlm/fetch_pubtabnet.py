"""Pin the upstream-linked full dataset and optionally resume its archive download."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from .io_utils import sha256_file, write_json


def download_requests(url: str, partial: Path, total: int, max_seconds: int) -> None:
    """Resume with verified byte offsets; keep normal TLS verification enabled."""
    import requests
    deadline = time.monotonic() + max_seconds
    with requests.Session() as session:
        while time.monotonic() < deadline:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset == total:
                return
            if offset > total:
                raise ValueError('Partial archive is larger than the pinned source')
            try:
                # A fresh query avoids reusing an expired CDN redirect on a resume.
                request_url = url + ('&' if '?' in url else '?') + f'download=true&resume={offset}'
                with session.get(request_url, headers={'Range': f'bytes={offset}-'}, stream=True,
                                 timeout=(20, 45)) as response:
                    response.raise_for_status()
                    if response.status_code == 206:
                        expected = f'bytes {offset}-'
                        if not response.headers.get('Content-Range', '').startswith(expected):
                            raise ValueError('Resume response has an incorrect offset')
                    elif response.status_code != 200 or offset:
                        raise ValueError('Server did not honor the resume range')
                    with partial.open('ab') as stream:
                        last_print = time.monotonic()
                        for block in response.iter_content(1024*1024):
                            if offset + len(block) > total:
                                raise ValueError('Response exceeds pinned archive size')
                            stream.write(block)
                            offset += len(block)
                            if time.monotonic()-last_print >= 15:
                                print(json.dumps({'downloaded_bytes': offset, 'total_bytes': total,
                                                  'percent': round(offset/total*100, 2)}), flush=True)
                                last_print = time.monotonic()
                            if time.monotonic() > deadline:
                                raise TimeoutError('Bounded download attempt ended; resume the same file')
            except requests.RequestException as exc:
                print(json.dumps({'retry': type(exc).__name__, 'partial_bytes':
                                  partial.stat().st_size if partial.exists() else 0}), flush=True)
                time.sleep(2)
                continue
    raise TimeoutError('Download time limit reached; rerun to resume')


def main() -> None:
    from huggingface_hub import HfApi

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-record', type=Path, default=Path('research/pubtabnet_full_source.json'))
    parser.add_argument('--output-dir', type=Path, default=Path('data/raw/pubtabnet-full'))
    parser.add_argument('--download', action='store_true', help='Without this flag, only inspect/pin metadata')
    parser.add_argument('--transport', choices=['requests', 'curl'], default='requests')
    parser.add_argument('--max-seconds', type=int, default=900, help='Bound each resumable download attempt')
    args = parser.parse_args()
    repo = 'ajimeno/PubTabNet'
    if args.source_record.exists():
        source = json.loads(args.source_record.read_text(encoding='utf-8'))
    else:
        info = HfApi().dataset_info(repo, files_metadata=True)
        item = next(s for s in info.siblings if s.rfilename == 'pubtabnet.tar.gz')
        if item.lfs is None or item.size is None:
            raise ValueError('No publisher file hash/size available for archive')
        source = {'repo': repo, 'revision': info.sha, 'filename': item.rfilename,
                  'bytes': item.size, 'lfs_sha256': item.lfs.sha256,
                  'url': f'https://huggingface.co/datasets/{repo}/resolve/{info.sha}/{item.rfilename}',
                  'metadata_license': 'cdla-sharing-1.0',
                  'origin': 'https://github.com/ibm-aur-nlp/PubTabNet',
                  'notes': ['Official upstream README points to this dataset repository.',
                            'GitHub annotation license and HF card tags differ; retain both (see DATA_POLICY.md).',
                            'Metadata receipt does not mean archive downloaded or contents inspected.']}
        write_json(args.source_record, source)
    if source.get('repo') != repo or source.get('filename') != 'pubtabnet.tar.gz':
        raise ValueError('Unexpected dataset identity')
    print(json.dumps({'metadata_pinned': True, 'revision': source['revision'], 'bytes': source['bytes'],
                      'download_requested': args.download}), flush=True)
    if not args.download:
        return
    if not 30 <= args.max_seconds <= 3600:
        raise ValueError('Download attempt must be bounded to 30..3600 seconds')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    final = args.output_dir/source['filename']
    partial = final.with_suffix(final.suffix+'.part')
    if final.exists():
        if final.stat().st_size != source['bytes'] or sha256_file(final) != source['lfs_sha256']:
            raise ValueError('Existing complete archive differs; do not overwrite it')
        print('Full archive already verified')
        return
    command = ['curl.exe' if os.name == 'nt' else 'curl', '--fail', '--location', '--progress-bar',
               '--connect-timeout', '20', '--speed-time', '60', '--speed-limit', '1024',
               '--max-time', str(args.max_seconds), '--continue-at', '-', '--output', str(partial), source['url']]
    completed = subprocess.run(command, check=False) if args.transport == 'curl' else None
    if args.transport == 'requests':
        download_requests(source['url'], partial, source['bytes'], args.max_seconds)
    if completed is not None and completed.returncode:
        print(json.dumps({'complete': False, 'partial_bytes': partial.stat().st_size if partial.exists() else 0,
                          'resume': 'Re-run the same command; never extract a .part file.'}))
        raise SystemExit(completed.returncode)
    if partial.stat().st_size != source['bytes'] or sha256_file(partial) != source['lfs_sha256']:
        raise ValueError('Archive checksum/size mismatch; never extract')
    partial.rename(final)
    write_json(args.output_dir/'source_receipt.json', source)
    print(json.dumps({'complete': True, 'verified_archive': str(final)}))


if __name__ == '__main__':
    main()
