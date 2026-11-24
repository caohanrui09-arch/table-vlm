"""Generate separate smoke/clean/augmented configs from the upstream-derived YAML."""

import argparse
import json
from pathlib import Path

import yaml

from .io_utils import read_jsonl, sha256_file
from .audit_dataset import verify_dataset


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=Path('models/Qwen3-VL-2B-Instruct'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true', help='Eight steps only; no effectiveness claim')
    parser.add_argument('--max-steps', type=int, default=8)
    args = parser.parse_args()
    if args.output.exists() or args.run_dir.exists():
        raise FileExistsError('Config and run directory must be new')
    audit = json.loads((args.data/'audit.json').read_text(encoding='utf-8'))
    if not args.smoke and audit['mode'] != 'official':
        raise ValueError('Refuse formal training config for smoke-only sample data')
    verify_dataset(args.data)
    for split in ['train', 'dev']:
        if sha256_file(args.data/f'{split}.jsonl') != audit['manifest_sha256'][split]:
            raise ValueError('Prepared data manifest differs from audit')
        if not read_jsonl(args.data/f'{split}.jsonl'):
            raise ValueError('Empty training/development data')
    config = yaml.safe_load((ROOT/'configs/lora_24gb.yaml').read_text(encoding='utf-8'))
    config.update(model_name_or_path=args.model.as_posix(), dataset_dir=(args.data/'lf').as_posix(),
                  media_dir=args.data.as_posix(), output_dir=args.run_dir.as_posix())
    config['overwrite_cache'] = True  # Do not reuse the 32768-token diagnostic encoding cache.
    if args.smoke:
        if not 1 <= args.max_steps <= 100:
            raise ValueError('Smoke step budget must be 1..100')
        config.update(max_steps=args.max_steps, gradient_accumulation_steps=1,
                      logging_steps=1, save_steps=args.max_steps, eval_strategy='no')
        config.pop('eval_steps', None)
        config.pop('eval_dataset', None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        yaml.safe_dump(config, stream, allow_unicode=True, sort_keys=False)
    print(json.dumps({'config': str(args.output), 'smoke': args.smoke, 'augmentation': audit['augmentation'],
                      'note': 'Configuration generated only; environment, preprocessing and training must still run.'}))


if __name__ == '__main__':
    main()
