"""Bind the completed LF encoding gate to one clean formal experiment."""

import argparse
import importlib.util
import json
import math
from pathlib import Path
import zipfile

import yaml

from .audit_dataset import verify_dataset
from .html_utils import NORMALIZATION_VERSION
from .io_utils import read_jsonl, sha256_file, unique_by_id, write_json


ROOT = Path(__file__).resolve().parents[1]
GATE = 'reports/lf-preflight-20261031T045740Z'


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def verify_gate(root: Path) -> dict:
    gate = root/GATE
    if read(gate/'exit-status.json')['exit_code'] != 0 or not read(gate/'ready.json')['passed']:
        raise ValueError('The actual cloud encoding gate did not pass')
    config_sha = sha256_file(root/'configs/lora_24gb.yaml')
    receipt_sha = sha256_file(root/'models/Qwen3-VL-2B-Instruct/source_receipt.json')
    fingerprint = {}
    for profile in ['formal-clean', 'formal-aug']:
        summary = read(gate/profile/'summary.json')
        if (not summary['passed'] or summary['failures'] != 0
                or summary['preflight_version'] != 'lf-sft-v3-nfc-exact-tokens'
                or summary['formal_config_sha256'] != config_sha
                or summary['model_receipt_sha256'] != receipt_sha):
            raise ValueError(f'Stale or failed encoding gate: {profile}')
        if read_jsonl(gate/profile/'failures.jsonl'):
            raise ValueError('Encoding failures were not empty')
        samples = read_jsonl(gate/profile/'samples.jsonl')
        encoded = unique_by_id(samples)
        if len(encoded) != 3140:
            raise ValueError('Encoding sample count changed')
        data = root/'data/prepared'/profile
        for split, count in [('train', 2990), ('dev', 150)]:
            rows = read_jsonl(data/f'{split}.jsonl')
            expected = summary['splits'][split]
            if len(rows) != count or expected['manifest_sha256'] != sha256_file(data/f'{split}.jsonl'):
                raise ValueError('Manifest changed after real LF encoding')
            for row in rows:
                item = encoded.get(row['id'], {})
                if (item.get('split') != split or not item.get('complete_target')
                        or not item.get('exact_answer_tokens') or not 0 < item.get('tokens', 0) <= 4096):
                    raise ValueError(f'Missing or invalid encoding record: {row["id"]}')
        fingerprint[profile] = {'summary_sha256': sha256_file(gate/profile/'summary.json'),
                                'samples_sha256': sha256_file(gate/profile/'samples.jsonl')}
    config = yaml.safe_load((root/'configs/lora_24gb.yaml').read_text(encoding='utf-8'))
    if (config['num_train_epochs'] != 2.0 or config.get('max_steps', -1) > 0
            or config.get('adapter_name_or_path') or config['cutoff_len'] != 4096
            or config['packing'] or config['image_max_pixels'] != 196608):
        raise ValueError('Formal configuration no longer matches the approved experiment')
    return {'passed': True, 'gate': GATE, 'config_sha256': config_sha,
            'model_receipt_sha256': receipt_sha, 'profiles': fingerprint}


def check_predictions(root: Path, run: Path, method: str) -> dict:
    manifest = root/'data/prepared/formal-clean/dev.jsonl'
    rows = unique_by_id(read_jsonl(manifest))
    path = run/method/'predictions.jsonl'
    predictions = unique_by_id(read_jsonl(path))
    metadata = read(path.with_suffix('.run.json'))
    if (set(rows) != set(predictions) or len(predictions) != 150
            or metadata['manifest_sha256'] != sha256_file(manifest)
            or metadata['max_new_tokens'] != 4096 or metadata['do_sample']):
        raise ValueError('Prediction cohort or decoding protocol differs')
    errors = [r['id'] for r in predictions.values() if r['status'] not in {'ok', 'truncated'}]
    if errors:
        raise ValueError(f'Runtime errors in {method}: {errors}; preserve outputs and inspect before continuing')
    return metadata


def summarize(root: Path, tag: str) -> None:
    run, report = root/'runs'/tag, root/'reports'/tag
    adapter = run/'q1-adapter'
    for name in ['adapter_model.safetensors', 'adapter_config.json', 'trainer_state.json', 'train_results.json']:
        if not (adapter/name).is_file():
            raise ValueError(f'Missing final training artifact: {name}')
    state, training = read(adapter/'trainer_state.json'), read(adapter/'train_results.json')
    if (state['global_step'] <= 8 or state['global_step'] != state['max_steps']
            or state['epoch'] < 1.999 or not math.isfinite(training['train_loss'])):
        raise ValueError('Two-epoch formal training did not complete')
    metrics, metadata = {}, {}
    for method in ['q0', 'q1']:
        metadata[method] = check_predictions(root, run, method)
        result = read(report/method/'summary.json')
        if result['samples'] != 150 or result['normalization_version'] != NORMALIZATION_VERSION:
            raise ValueError('Scoring protocol differs')
        metrics[method] = {k: result[k] for k in ['teds', 'teds_s', 'valid_html_rate', 'categories']}
    for key in ['model', 'revision', 'prompt', 'manifest_sha256', 'max_pixels', 'max_new_tokens', 'do_sample']:
        if metadata['q0'][key] != metadata['q1'][key]:
            raise ValueError(f'Q0/Q1 inference settings differ: {key}')
    if metadata['q0']['adapter'] is not None or Path(metadata['q1']['adapter']).resolve() != adapter.resolve():
        raise ValueError('Q1 did not use this run\'s final adapter')
    result = {'engineering_completed': True, 'dataset': 'local PubTabNet held-out dev 150',
              'test_evaluated': False, 'augmented_training_ran': False,
              'global_steps': state['global_step'], 'epoch': state['epoch'],
              'train_results': training, 'normalization_version': NORMALIZATION_VERSION,
              'metrics': metrics, 'delta_q1_minus_q0': {
                  k: metrics['q1'][k]-metrics['q0'][k] for k in ['teds', 'teds_s', 'valid_html_rate']},
              'note': 'Final checkpoint at two epochs, not best checkpoint selected on test; review raw cases.'}
    write_json(report/'comparison.json', result)
    print(json.dumps(result, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['verify', 'check-q0', 'check-q1', 'check-q2', 'summarize', 'package'])
    parser.add_argument('--tag', required=True)
    parser.add_argument('--exit-code', type=int, default=0)
    args = parser.parse_args()
    if '/' in args.tag or '\\' in args.tag or not args.tag.startswith(('formal-clean-', 'formal-aug-')):
        raise ValueError('Invalid run tag')
    report, run = ROOT/'reports'/args.tag, ROOT/'runs'/args.tag
    if args.action == 'verify':
        result = verify_gate(ROOT)
        result['project_source_sha256'] = {name: sha256_file(ROOT/name) for name in [
            'table_vlm/formal_protocol.py', 'table_vlm/training_config.py',
            'table_vlm/predict.py', 'table_vlm/evaluate.py', 'table_vlm/html_utils.py',
            'scripts/cloud_formal_clean.sh', 'table_vlm/aug_comparison.py',
            'scripts/cloud_formal_aug.sh'] if (ROOT/name).is_file()}
        verify_dataset(ROOT/'data/prepared/formal-clean')
        # Recheck current source and every model file, not just a success receipt.
        spec = importlib.util.find_spec('llamafactory')
        if spec is None or not spec.origin:
            raise ValueError('Installed LlamaFactory not found')
        for name, expected in read(ROOT/'research/lf-preflight-source.json')['files'].items():
            if sha256_file(Path(spec.origin).parent/name) != expected:
                raise ValueError(f'Installed LF source changed: {name}')
        model = ROOT/'models/Qwen3-VL-2B-Instruct'
        for item in read(ROOT/'research/qwen2b-fixed-manifest.json')['files']:
            path = (model/item['path']).resolve()
            if not path.is_relative_to(model.resolve()) or sha256_file(path) != item['sha256']:
                raise ValueError(f'Fixed model file changed: {item["path"]}')
        write_json(report/'gate-binding.json', result)
        print('FORMAL_GATE_VERIFIED')
    elif args.action.startswith('check-'):
        check_predictions(ROOT, run, args.action.removeprefix('check-'))
    elif args.action == 'summarize':
        summarize(ROOT, args.tag)
    else:
        report.mkdir(parents=True, exist_ok=True)
        write_json(report/'exit-status.json', {'exit_code': args.exit_code,
            'completed': args.exit_code == 0 and (report/'comparison.json').is_file(),
            'test_evaluated': False})
        destination = ROOT/f'{args.tag}-results.zip'
        with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
            for folder in [report, run]:
                if folder.exists():
                    for p in sorted(folder.rglob('*')):
                        if p.is_file() and p.suffix in {'.json', '.jsonl', '.safetensors', '.md', '.jinja', '.txt'}:
                            archive.write(p, p.relative_to(ROOT).as_posix())
            for p in [ROOT/'logs'/f'{args.tag}.log', ROOT/'configs/generated'/f'{args.tag}.yaml']:
                if p.is_file():
                    archive.write(p, p.relative_to(ROOT).as_posix())
        print(f'RESULTS_ZIP={destination}', flush=True)


if __name__ == '__main__':
    main()
