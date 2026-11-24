"""Compare an equal-exposure augmented SFT run with the completed clean run."""

import argparse
import json
import math
from pathlib import Path

import yaml

from .formal_protocol import ROOT, check_predictions, read
from .html_utils import NORMALIZATION_VERSION
from .io_utils import sha256_file, write_json


CLEAN = 'formal-clean-20261031T051007Z'


def verify_config_pair(clean: dict, augmented: dict) -> None:
    allowed = {'dataset_dir', 'media_dir', 'output_dir'}
    if {k:v for k,v in clean.items() if k not in allowed} != {
            k:v for k,v in augmented.items() if k not in allowed}:
        raise ValueError('Q1/Q2 hyperparameters differ beyond dataset/output paths')
    if ('formal-clean' not in clean['dataset_dir'] or 'formal-aug' not in augmented['dataset_dir']
            or augmented.get('adapter_name_or_path') or augmented['num_train_epochs'] != 2.0):
        raise ValueError('Q2 must train the augmented dataset from the original model')


def verify_clean_reference(tag: str) -> dict:
    original = ROOT/'reports'/CLEAN
    if not read(original/'exit-status.json')['completed']:
        raise ValueError('Completed clean reference is required')
    clean_config = yaml.safe_load((ROOT/'configs/generated'/f'{CLEAN}.yaml').read_text(encoding='utf-8'))
    aug_config = yaml.safe_load((ROOT/'configs/generated'/f'{tag}.yaml').read_text(encoding='utf-8'))
    verify_config_pair(clean_config, aug_config)
    summaries = {}
    for name in ['q0', 'q1']:
        check_predictions(ROOT, ROOT/'runs'/CLEAN, name)
        summary = read(original/name/'summary.json')
        if summary['samples'] != 150 or summary['normalization_version'] != NORMALIZATION_VERSION:
            raise ValueError('Clean reference scoring protocol changed')
        path = ROOT/'runs'/CLEAN/name/'predictions.jsonl'
        if summary['predictions_sha256'] != sha256_file(path):
            raise ValueError('Clean reference predictions changed')
        summaries[name] = sha256_file(original/name/'summary.json')
    return {'clean_reference': CLEAN, 'clean_summary_sha256': summaries,
            'clean_adapter_sha256': sha256_file(ROOT/'runs'/CLEAN/'q1-adapter/adapter_model.safetensors'),
            'equal_hyperparameters': True, 'original_model_initialization': True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['verify', 'summarize'])
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    if '/' in args.tag or '\\' in args.tag or not args.tag.startswith('formal-aug-'):
        raise ValueError('Invalid augmented run tag')
    report, run = ROOT/'reports'/args.tag, ROOT/'runs'/args.tag
    reference = verify_clean_reference(args.tag)
    if args.action == 'verify':
        write_json(report/'clean-reference.json', reference)
        print('Q2_REFERENCE_VERIFIED: same hyperparameters; clean predictions reused, not regenerated.')
        return
    adapter = run/'q2-adapter'
    for name in ['adapter_config.json','adapter_model.safetensors','trainer_state.json','train_results.json']:
        if not (adapter/name).is_file():
            raise ValueError(f'Missing final augmented artifact: {name}')
    state, training = read(adapter/'trainer_state.json'), read(adapter/'train_results.json')
    clean = read(ROOT/'reports'/CLEAN/'comparison.json')
    if (state['global_step'] != clean['global_steps'] or state['global_step'] != state['max_steps']
            or state['epoch'] < 1.999 or not math.isfinite(training['train_loss'])):
        raise ValueError('Equal-exposure augmented training did not complete')
    meta = check_predictions(ROOT, run, 'q2')
    old_meta = read(ROOT/'runs'/CLEAN/'q1/predictions.run.json')
    for key in ['model','revision','manifest_sha256','prompt','max_pixels','max_new_tokens','do_sample']:
        if meta[key] != old_meta[key]:
            raise ValueError(f'Inference protocol differs: {key}')
    if Path(meta['adapter']).resolve() != adapter.resolve():
        raise ValueError('Q2 did not use the final augmented adapter')
    current = read(report/'q2/summary.json')
    if current['samples'] != 150 or current['normalization_version'] != NORMALIZATION_VERSION:
        raise ValueError('Augmented scoring protocol changed')
    metrics = {**clean['metrics'], 'q2': {k: current[k] for k in
               ['teds','teds_s','valid_html_rate','categories']}}
    result = {'engineering_completed': True, 'clean_reference': reference,
        'dataset': 'same clean local PubTabNet dev150', 'test_evaluated': False,
        'degraded_input_robustness_evaluated': False, 'global_steps': state['global_step'],
        'epoch': state['epoch'], 'train_results': training, 'metrics': metrics,
        'delta_q2_minus_q1': {k: metrics['q2'][k]-metrics['q1'][k] for k in ['teds','teds_s','valid_html_rate']},
        'normalization_version': NORMALIZATION_VERSION,
        'note': 'Report negative results as measured; clean-dev scoring alone does not prove degraded-input robustness.'}
    write_json(report/'comparison.json', result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
