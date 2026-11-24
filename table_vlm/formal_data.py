"""Select a reproducible, paper-separated subset from the verified PubTabNet archive.

Reuses the existing annotation converter, exporter and ImageHash implementations.
Only annotations and a bounded candidate image pool are extracted, all to --root.
"""

import argparse
import hashlib
import heapq
import json
import shutil
import tarfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

from PIL import Image

from .audit_dataset import verify_dataset
from .data import _key, allocate, document_id, prepare
from .html_utils import annotation_html
from .io_utils import iter_jsonl, sha256_file, write_json, write_jsonl


def safe_member(member: tarfile.TarInfo) -> PurePosixPath:
    name = PurePosixPath(member.name)
    if name.is_absolute() or '..' in name.parts or '\\' in member.name or ':' in member.name:
        raise ValueError(f'Unsafe archive path: {member.name}')
    if not member.isdir() and not member.isfile():
        raise ValueError(f'Unsupported archive entry: {member.name}')
    return name


def extract_annotations(archive: Path, root: Path, source: dict) -> Path:
    receipt_path = root/'raw/archive-inspection.json'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        path = root/receipt['annotation']
        if receipt['archive_sha256'] != source['lfs_sha256'] or sha256_file(path) != receipt['annotation_sha256']:
            raise ValueError('Archive inspection receipt differs')
        return path
    if archive.stat().st_size != source['bytes'] or sha256_file(archive) != source['lfs_sha256']:
        raise ValueError('Archive must match the pinned upstream size and SHA256 before extraction')
    destination = root/'raw/annotations'
    destination.mkdir(parents=True, exist_ok=True)
    annotations = []
    counts, sizes = Counter(), Counter()
    with tarfile.open(archive, mode='r|gz') as tar:
        for n, member in enumerate(tar, 1):
            name = safe_member(member)
            if not member.isfile():
                continue
            suffix = name.suffix.lower()
            counts[suffix] += 1
            sizes[suffix] += member.size
            if suffix == '.jsonl':
                if member.size > 10_000_000_000:
                    raise ValueError('Annotation exceeds declared extraction limit')
                path = destination/name.name
                with tar.extractfile(member) as stream, path.with_suffix('.jsonl.part').open('xb') as output:
                    shutil.copyfileobj(stream, output, 1024*1024)
                path.with_suffix('.jsonl.part').rename(path)
                annotations.append(path)
            if n % 100000 == 0:
                print(json.dumps({'stage': 'archive_inspection', 'members': n}), flush=True)
    if len(annotations) != 1:
        raise ValueError(f'Expected exactly one annotation JSONL, found {len(annotations)}')
    path = annotations[0]
    write_json(receipt_path, {'archive_sha256': source['lfs_sha256'], 'archive_bytes': source['bytes'],
        'annotation': path.relative_to(root).as_posix(), 'annotation_sha256': sha256_file(path),
        'file_counts': dict(counts), 'uncompressed_sizes': dict(sizes),
        'note': 'Scanned entry names/types, extracted annotations only. No official test image inspected.'})
    return path


def select_candidates(annotations: Path, root: Path, counts: dict[str, int], seed: int) -> list[dict]:
    receipt_path = root/'raw/candidate-selection.json'
    candidate_path = root/'raw/candidate-annotations.jsonl'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        if receipt['seed'] != seed or receipt['targets'] != counts or receipt['annotation_sha256'] != sha256_file(annotations):
            raise ValueError('Candidate selection identity changed; use a new root')
        if sha256_file(candidate_path) != receipt['candidate_sha256']:
            raise ValueError('Candidate file changed')
        return list(iter_jsonl(candidate_path))
    heaps: dict[str, list] = {s: [] for s in counts}
    source_papers: dict[str, set[str]] = defaultdict(set)
    source_counts = Counter()
    for n, row in enumerate(iter_jsonl(annotations), 1):
        source = row.get('split')
        source_counts[str(source)] += 1
        if source not in {'train', 'val'}:
            continue
        paper = document_id(row['filename'])
        source_papers[paper].add(source)
        if n % 100000 == 0:
            print(json.dumps({'stage': 'source_split_scan', 'rows': n}), flush=True)
    crossing = sorted(p for p, sources in source_papers.items() if len(sources) > 1)
    crossing_set = set(crossing)
    # Resolve source-paper leakage before ranking, so quarantined papers do not consume reserve slots.
    for n, row in enumerate(iter_jsonl(annotations), 1):
        source = row.get('split')
        if source not in {'train', 'val'}:
            continue
        paper = document_id(row['filename'])
        if paper in crossing_set:
            continue
        split = 'train' if source == 'train' else ('dev' if int(_key(seed, paper)[:8], 16) % 2 == 0 else 'test')
        rank = int(_key(seed, row['filename']), 16)
        heap = heaps[split]
        # The filename tie-breaker avoids comparing dictionaries on hash collision.
        item = (-rank, row['filename'], row)
        if len(heap) < counts[split]*3:
            heapq.heappush(heap, item)
        elif item[:2] > heap[0][:2]:
            heapq.heapreplace(heap, item)
        if n % 100000 == 0:
            print(json.dumps({'stage': 'seeded_candidate_scan', 'rows': n}), flush=True)
    candidates = [item[2] for split in ['dev', 'test', 'train']
                  for item in sorted(heaps[split], reverse=True)
                  if document_id(item[2]['filename']) not in crossing_set]
    write_jsonl(candidate_path, candidates)
    write_json(receipt_path, {'seed': seed, 'targets': counts, 'reserve_factor': 3,
        'sampling': 'smallest seed/filename SHA256 ranks over full metadata; not first-N',
        'annotation_sha256': sha256_file(annotations), 'source_counts': dict(source_counts),
        'source_papers': len(source_papers), 'cross_source_papers_quarantined': crossing,
        'candidates': len(candidates), 'candidate_sha256': sha256_file(candidate_path)})
    return candidates


def extract_candidates(archive: Path, root: Path, candidates: list[dict]) -> Path:
    base = root/'raw/candidate-images'
    targets = {f"{r['split']}/{r['filename']}" for r in candidates}
    receipt_path = root/'raw/candidate-images.json'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        if receipt['requested'] != sorted(targets):
            raise ValueError('Candidate image request changed')
        return base
    extracted = set()
    base.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, mode='r|gz') as tar:
        for n, member in enumerate(tar, 1):
            name = safe_member(member)
            if not member.isfile() or len(name.parts) < 2:
                continue
            key = '/'.join(name.parts[-2:])
            if key not in targets:
                continue
            if key in extracted or member.size > 32*1024*1024:
                raise ValueError(f'Duplicate or oversized image member: {key}')
            path = base/key
            path.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as stream, path.open('xb') as output:
                shutil.copyfileobj(stream, output)
            extracted.add(key)
            if len(extracted) % 1000 == 0:
                print(json.dumps({'stage': 'candidate_images', 'extracted': len(extracted)}), flush=True)
    write_json(receipt_path, {'requested': sorted(targets), 'extracted': len(extracted),
                             'missing': sorted(targets-extracted)})
    return base


def choose_subset(candidates: list[dict], images: Path, root: Path, counts: dict[str, int], seed: int) -> Path:
    import imagehash
    output = root/'raw/selected-annotations.jsonl'
    if output.exists():
        raise FileExistsError('Selection already exists; preserve it and use the prepare-only stage')
    mapping = allocate(candidates, 'official', seed)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in candidates:
        grouped[mapping[document_id(row['filename'])]].append(row)
    accepted, rejected, near_pairs = [], [], []
    fingerprints, pixel_hashes = {}, set()
    retained = []
    for split in ['dev', 'test', 'train']:
        papers = set()
        taken = 0
        for row in sorted(grouped[split], key=lambda r: _key(seed, r['filename'])):
            if taken == counts[split]:
                break
            name, reason = row['filename'], ''
            paper = document_id(name)
            if paper in papers:
                rejected.append({'id': Path(name).stem, 'split': split, 'reason': 'one_table_per_paper'})
                continue
            path = images/row['split']/name
            try:
                target = annotation_html(row)
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    rgb = image.convert('RGB')
                    rgb.load()
                byte_hash = sha256_file(path)
                pixel_hash = hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
                phash, dhash = int(str(imagehash.phash(rgb)), 16), int(str(imagehash.dhash(rgb)), 16)
                aspect = rgb.width/rgb.height
                if byte_hash in fingerprints or pixel_hash in pixel_hashes:
                    reason = 'exact_bytes_or_decoded_pixels_duplicate'
                else:
                    # These are conservative cross-split quarantine candidates, not confirmed semantic duplicates.
                    for old in retained:
                        if old['split'] == split or abs(aspect/old['aspect']-1) > .05:
                            continue
                        pd, dd = (phash ^ old['phash']).bit_count(), (dhash ^ old['dhash']).bit_count()
                        if pd <= 4 and dd <= 4:
                            near_pairs.append({'id': Path(name).stem, 'split': split, 'other_id': old['id'],
                                'other_split': old['split'], 'phash_distance': pd, 'dhash_distance': dd,
                                'decision': 'quarantined_before_predictions', 'confirmed_duplicate': False})
                            reason = 'cross_split_near_duplicate_candidate'
                            break
                if not reason:
                    fingerprints[byte_hash] = split
                    pixel_hashes.add(pixel_hash)
                    retained.append({'id': Path(name).stem, 'split': split, 'phash': phash,
                                     'dhash': dhash, 'aspect': aspect})
                    accepted.append(row)
                    papers.add(paper)
                    taken += 1
            except (OSError, ValueError, KeyError, TypeError) as exc:
                reason = f'{type(exc).__name__}: {exc}'
            if reason:
                rejected.append({'id': Path(name).stem, 'split': split, 'reason': reason})
            if taken and taken % 250 == 0:
                print(json.dumps({'stage': 'quality_selection', 'split': split, 'retained': taken}), flush=True)
        if taken != counts[split]:
            raise ValueError(f'Insufficient valid independent candidates: {split}={taken}/{counts[split]}; expand reserve explicitly')
    write_jsonl(output, accepted)
    write_jsonl(root/'reports/data/rejected-candidates.jsonl', rejected)
    write_jsonl(root/'reports/data/near-duplicate-candidates.jsonl', near_pairs)
    write_json(root/'reports/data/selection.json', {'seed': seed, 'split_counts': counts,
        'selected_sha256': sha256_file(output), 'candidates_in_reserve': len(candidates),
        'rejected_examined': len(rejected), 'rejection_counts': dict(Counter(r['reason'].split(':')[0] for r in rejected)),
        'near_duplicate_candidate_pairs': len(near_pairs), 'near_duplicate_algorithm': 'ImageHash4.3.2 pHash64<=4 AND dHash64<=4 AND aspect ratio difference<=5%, cross-split quarantine',
        'imagehash_version': imagehash.__version__, 'sampling': 'seeded full-metadata rank; one table per paper; dev then test then train',
        'limitations': ['Only retained/reserve candidates image-audited, not all source images.',
            'Not an official leaderboard split or a representative population estimate.',
            'Perceptual hashes cannot prove absence of semantic duplicates or pretraining contamination.',
            'HTML syntax/cell alignment validated; image-to-label semantic correctness still needs visual review.']})
    return output


def export_prepared(root: Path, seed: int) -> None:
    selected = root/'raw/selected-annotations.jsonl'
    images = root/'raw/candidate-images'
    for label, augment in [('formal-clean-v1', False), ('formal-aug-v1', True)]:
        output = root/'prepared'/label
        if output.exists():
            raise FileExistsError(output)
        receipt = prepare(selected, images, output, 'official', seed, augment, 100000, 100000)
        receipt['selection_report'] = '../../reports/data/selection.json'
        receipt['selection_report_sha256'] = sha256_file(root/'reports/data/selection.json')
        receipt['near_duplicate_check'] = 'Cross-split ImageHash pHash/dHash candidate quarantine; see selection report'
        receipt['limitations'] = ['Seeded curated subset with one table per paper, not an official leaderboard.',
                                 'Near-duplicate detection is heuristic; pretraining contamination unknown.']
        # Replace only this newly generated receipt to add upstream-selection provenance.
        (output/'audit.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        audit = verify_dataset(output, root/'prepared/formal-clean-v1' if augment else None)
        write_json(root/f'reports/data/{label}-audit.json', audit)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source-record', type=Path, default=Path('research/pubtabnet_full_source.json'))
    parser.add_argument('--train', type=int, default=3000)
    parser.add_argument('--dev', type=int, default=150)
    parser.add_argument('--test', type=int, default=150)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if min(args.train, args.dev, args.test) < 1:
        raise ValueError('Each split must be nonempty')
    root = args.root.resolve()
    if not args.prepare_only:
        source = json.loads(args.source_record.read_text(encoding='utf-8'))
        annotations = extract_annotations(args.archive, root, source)
        candidates = select_candidates(annotations, root, {'train': args.train, 'dev': args.dev, 'test': args.test}, args.seed)
        images = extract_candidates(args.archive, root, candidates)
        choose_subset(candidates, images, root, {'train': args.train, 'dev': args.dev, 'test': args.test}, args.seed)
    export_prepared(root, args.seed)


if __name__ == '__main__':
    main()
