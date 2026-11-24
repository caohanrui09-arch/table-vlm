"""Small, strict JSON utilities shared by command-line adapters."""

import hashlib
import json
from pathlib import Path
from collections.abc import Iterator


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding='utf-8-sig') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f'{path}:{line_number}: expected a JSON object')
            yield row


def read_jsonl(path: Path) -> list[dict]:
    return list(iter_jsonl(path))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n')


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')


def unique_by_id(rows: list[dict]) -> dict[str, dict]:
    result = {}
    for row in rows:
        sample_id = row.get('id')
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError('Every row requires a nonempty string id')
        if sample_id in result:
            raise ValueError(f'Duplicate id: {sample_id}')
        result[sample_id] = row
    return result

