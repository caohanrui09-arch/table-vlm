"""Audit real, untruncated LlamaFactory SFT encoding on CPU; never load weights."""

import argparse
import importlib.metadata
import json
import os
import unicodedata
from pathlib import Path

from .io_utils import read_jsonl, sha256_file, write_json, write_jsonl
from .prompts import PROMPT


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = 'qwen3_vl_nothink'
DIAGNOSTIC_CUTOFF = 32768
PREFLIGHT_VERSION = 'lf-sft-v3-nfc-exact-tokens'


def check_encoded(row: dict, target: str, tokenizer, cutoff: int) -> dict:
    """Reject masked answers, prompt leakage, lost image tokens and truncation."""
    ids, labels = row['input_ids'], row['labels']
    if not ids or len(ids) != len(labels) or row['attention_mask'] != [1] * len(ids):
        raise ValueError('Malformed encoded sequence')
    supervised = [i for i, value in enumerate(labels) if value != -100]
    if not supervised or supervised != list(range(supervised[0], len(ids))):
        raise ValueError('Expected a masked prompt followed by a complete supervised answer')
    if any(labels[i] != ids[i] for i in supervised):
        raise ValueError('Supervised labels differ from input tokens')
    decoded = tokenizer.decode([labels[i] for i in supervised], skip_special_tokens=True,
                               clean_up_tokenization_spaces=False).strip()
    # Qwen's pinned tokenizer uses NFC. Canonically equivalent accented letters
    # must not be mistaken for target loss. No NFKC, whitespace or HTML repair.
    normalized_target = unicodedata.normalize('NFC', target)
    if decoded != normalized_target.strip():
        raise ValueError('Decoded supervised answer differs from complete exported HTML')
    prefix = tokenizer.decode(ids[:supervised[0]], skip_special_tokens=False,
                              clean_up_tokenization_spaces=False)
    if PROMPT not in prefix:
        raise ValueError('Expected instruction missing from masked prompt')
    image_token = tokenizer.convert_tokens_to_ids('<|image_pad|>')
    image_positions = [i for i, value in enumerate(ids) if value == image_token]
    if not image_positions or any(labels[i] != -100 for i in image_positions):
        raise ValueError('Image tokens absent or incorrectly supervised')
    eos = tokenizer.convert_tokens_to_ids('<|im_end|>')
    if eos not in [labels[i] for i in supervised]:
        raise ValueError('Complete assistant end token missing')
    # Exact assistant formatter in the pinned qwen3_vl_nothink template.
    # This also catches omitted/added special tokens hidden by decoded text.
    expected_ids = tokenizer.encode(target + '<|im_end|>\n', add_special_tokens=False)
    if [labels[i] for i in supervised] != expected_ids:
        raise ValueError('Supervised token sequence differs from complete formatted answer')
    if len(ids) >= DIAGNOSTIC_CUTOFF:
        raise ValueError('Diagnostic cap reached; full length is unknown')
    if len(ids) > cutoff:
        raise ValueError(f'Full sequence length {len(ids)} exceeds training cutoff {cutoff}')
    return {'tokens': len(ids), 'supervised_tokens': len(supervised),
            'image_tokens': len(image_positions), 'complete_target': True,
            'nfc_target_changed': normalized_target != target, 'exact_answer_tokens': True}


def main() -> None:
    from .audit_dataset import verify_dataset
    from .length_audit import distribution
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    # Set before importing torch/Transformers/LlamaFactory, including when invoked directly.
    os.environ.update(CUDA_VISIBLE_DEVICES='-1', HF_HUB_OFFLINE='1',
                      HF_DATASETS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    import torch
    import llamafactory
    from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
    from llamafactory.hparams import DataArguments, ModelArguments
    from llamafactory.model import load_tokenizer
    from transformers import Seq2SeqTrainingArguments

    torch.set_num_threads(4)
    lf_root = Path(llamafactory.__file__).parent
    source_pin = json.loads((ROOT/'research/lf-preflight-source.json').read_text(encoding='utf-8'))
    for name, expected in source_pin['files'].items():
        if sha256_file(lf_root/name) != expected:
            raise ValueError(f'LlamaFactory source differs from pinned commit: {name}')
    data, model = args.data.resolve(), args.model.resolve()
    verify_dataset(data)
    config = __import__('yaml').safe_load((ROOT/'configs/lora_24gb.yaml').read_text(encoding='utf-8'))
    if config['template'] != TEMPLATE or config['packing'] or config['cutoff_len'] != 4096:
        raise ValueError('Preflight no longer matches formal training configuration')
    args.output.mkdir(parents=True)
    model_args = ModelArguments(model_name_or_path=str(model), trust_remote_code=False,
                                image_max_pixels=config['image_max_pixels'],
                                image_min_pixels=config['image_min_pixels'],
                                cache_dir=str(args.output/'cache'))
    data_args = DataArguments(template=TEMPLATE, dataset='table_train', eval_dataset='table_dev',
                              dataset_dir=str(data/'lf'), media_dir=str(data),
                              cutoff_len=DIAGNOSTIC_CUTOFF, packing=False, overwrite_cache=True,
                              preprocessing_num_workers=1, preprocessing_batch_size=16)
    training_args = Seq2SeqTrainingArguments(output_dir=str(args.output/'unused-trainer'),
        do_train=True, use_cpu=True, bf16=False, fp16=False, report_to=[],
        logging_strategy='no', disable_tqdm=True, seed=42, dataloader_num_workers=0)
    module = load_tokenizer(model_args)
    if str(module['tokenizer'].backend_tokenizer.normalizer) != 'NFC()':
        raise ValueError('Pinned Qwen tokenizer NFC normalizer changed')
    if module['processor'] is None:
        raise ValueError('Multimodal processor was not loaded')
    template = get_template_and_fix_tokenizer(module['tokenizer'], data_args)
    datasets = get_dataset(template, model_args, data_args, training_args, stage='sft', **module)
    records, failures, split_summary = [], [], {}
    for split, key in [('train', 'train_dataset'), ('dev', 'eval_dataset')]:
        manifest = read_jsonl(data/f'{split}.jsonl')
        encoded = datasets[key]
        if encoded is None or len(encoded) != len(manifest):
            raise ValueError(f'{split}: encoded count differs from manifest (samples dropped)')
        lengths = []
        for expected, row in zip(manifest, encoded, strict=True):
            lengths.append(len(row['input_ids']))
            try:
                if row['images'] != [str((data/expected['image']).resolve())]:
                    raise ValueError('Encoded image/order differs from manifest')
                target = expected['html'].removeprefix('<html><body>').removesuffix('</body></html>')
                details = check_encoded(row, target, module['tokenizer'], config['cutoff_len'])
                records.append({'id': expected['id'], 'split': split, **details})
            except ValueError as error:
                failures.append({'id': expected['id'], 'split': split, 'tokens': len(row['input_ids']),
                                 'error': str(error)})
        split_summary[split] = {'samples': len(manifest), 'tokens': distribution(lengths),
                               'manifest_sha256': sha256_file(data/f'{split}.jsonl')}
    report = {'passed': not failures, 'preflight_version': PREFLIGHT_VERSION,
              'target_comparison': 'NFC text plus exact complete assistant token sequence',
              'nfc_targets': sum(r['nfc_target_changed'] for r in records),
              'training_ran': False, 'model_weights_loaded': False,
              'device': 'cpu', 'test_encoded': False, 'template': TEMPLATE,
              'image_min_pixels': config['image_min_pixels'], 'image_max_pixels': config['image_max_pixels'],
              'training_cutoff': config['cutoff_len'], 'diagnostic_cutoff': DIAGNOSTIC_CUTOFF,
              'splits': split_summary, 'failures': len(failures), 'lf_source': source_pin,
              'model_receipt_sha256': sha256_file(model/'source_receipt.json'),
              'formal_config_sha256': sha256_file(ROOT/'configs/lora_24gb.yaml'),
              'versions': {name: importlib.metadata.version(name) for name in
                           ['llamafactory', 'transformers', 'torch', 'datasets']}}
    write_jsonl(args.output/'samples.jsonl', records)
    write_jsonl(args.output/'failures.jsonl', failures)
    write_json(args.output/'summary.json', report)
    print(json.dumps(report))
    if failures:
        raise SystemExit('LF_PREFLIGHT_FAILED: inspect failures.jsonl; do not train or drop held-out samples')
    print('LF_PREFLIGHT_OK')


if __name__ == '__main__':
    main()
