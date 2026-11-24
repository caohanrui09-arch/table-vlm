"""Join CPU shard results only when every manifest sample is present exactly once."""

import argparse
import json
from pathlib import Path

from .io_utils import read_jsonl, sha256_file, unique_by_id, write_json, write_jsonl


def merge(manifest: Path, inputs: list[Path], output: Path) -> None:
    truth = unique_by_id(read_jsonl(manifest))
    records, runs = [], []
    for path in inputs:
        run = json.loads(path.with_suffix('.run.json').read_text(encoding='utf-8'))
        if run['manifest_sha256'] != sha256_file(manifest) or run['device'] != 'cpu' or run['compiled_with_cuda']:
            raise ValueError('Shard identity/device differs')
        if runs and any(run[key] != runs[0][key] for key in ['config_sha256', 'versions', 'models', 'cpu_threads', 'shards']):
            raise ValueError('Shard model/configuration differs')
        runs.append(run)
        records.extend(read_jsonl(path))
    actual = unique_by_id(records)
    if set(actual) != set(truth):
        raise ValueError('Shards have missing or unexpected sample IDs')
    write_jsonl(output, [actual[key] for key in truth])
    write_json(output.with_suffix('.run.json'), {'device': 'cpu', 'samples': len(records), 'shard_runs': runs,
        'manifest_sha256': sha256_file(manifest), 'predictions_sha256': sha256_file(output),
        'note': 'Concurrent CPU workers, four threads per worker. Per-image latency includes CPU contention; not isolated speed benchmarks.'})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    merge(args.manifest, args.inputs, args.output)


if __name__ == '__main__':
    main()
