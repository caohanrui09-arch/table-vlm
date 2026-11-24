"""CPU-only, resumable native Qwen processor length audit; never truncate targets."""

import argparse
import json
import os
from pathlib import Path

from PIL import Image

from .io_utils import read_jsonl, sha256_file, unique_by_id, write_json
from .prompts import PROMPT


def distribution(values: list[int]) -> dict:
    ordered = sorted(values)
    if not ordered:
        return {}
    return {**{f'p{p}': ordered[round((len(ordered)-1)*p/100)] for p in [0, 50, 90, 95, 99, 100]},
            'mean': sum(ordered)/len(ordered)}


def main() -> None:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['HF_HUB_OFFLINE'] = '1'
    import torch
    from transformers import AutoProcessor
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-pixels', type=int, default=196608)
    parser.add_argument('--cutoff-len', type=int, default=4096)
    args = parser.parse_args()
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True, trust_remote_code=False,
                                             min_pixels=4096, max_pixels=args.max_pixels)
    manifests = {s: args.data/f'{s}.jsonl' for s in ['train', 'dev', 'test']}
    binding = {'manifest_sha256': {s: sha256_file(p) for s, p in manifests.items()},
               'max_pixels': args.max_pixels, 'cutoff_len': args.cutoff_len,
               'model_receipt_sha256': sha256_file(args.model/'source_receipt.json'),
               'prompt': PROMPT, 'device': 'cpu'}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    binding_path = args.output_dir/'binding.json'
    if binding_path.exists():
        if json.loads(binding_path.read_text(encoding='utf-8')) != binding:
            raise ValueError('Length audit configuration changed; use a new output directory')
    else:
        write_json(binding_path, binding)
    checkpoint = args.output_dir/'lengths.jsonl'
    completed = unique_by_id(read_jsonl(checkpoint)) if checkpoint.exists() else {}
    records = [r for p in manifests.values() for r in read_jsonl(p)]
    if set(completed)-{r['id'] for r in records}:
        raise ValueError('Checkpoint has unexpected sample IDs')
    if (args.output_dir/'summary.json').exists():
        summary = json.loads((args.output_dir/'summary.json').read_text(encoding='utf-8'))
        if set(completed) != {r['id'] for r in records} or summary['lengths_sha256'] != sha256_file(checkpoint):
            raise ValueError('Completed length audit was changed')
        print(json.dumps({'already_complete': True, 'samples': len(completed)}), flush=True)
        return
    with checkpoint.open('a', encoding='utf-8', newline='\n') as stream, torch.inference_mode():
        for index, row in enumerate(records, 1):
            if row['id'] in completed:
                continue
            image_path = (args.data/row['image']).resolve()
            if not image_path.is_relative_to(args.data.resolve()) or sha256_file(image_path) != row['image_sha256']:
                raise ValueError('Input image identity failed')
            with Image.open(image_path) as source:
                image = source.convert('RGB')
                image.load()
            answer = row['html'].removeprefix('<html><body>').removesuffix('</body></html>')
            user = {'role': 'user', 'content': [{'type': 'image', 'image': image}, {'type': 'text', 'text': PROMPT}]}
            encoded = processor.apply_chat_template([user, {'role': 'assistant', 'content': [{'type': 'text', 'text': answer}]}],
                tokenize=True, return_dict=True, return_tensors='pt', add_generation_prompt=False)
            tokens = int(encoded['input_ids'].shape[-1])
            result = {'id': row['id'], 'split': row['split'], 'width': image.width, 'height': image.height,
                      'html_characters': len(answer), 'target_text_tokens': len(processor.tokenizer.encode(answer, add_special_tokens=False)),
                      'multimodal_tokens': tokens, 'image_grid_thw': encoded['image_grid_thw'].tolist(),
                      'exceeds_cutoff': tokens > args.cutoff_len}
            stream.write(json.dumps(result)+'\n')
            stream.flush()
            completed[row['id']] = result
            if index % 50 == 0:
                print(json.dumps({'stage': 'length_audit', 'completed': index, 'total': len(records)}), flush=True)
    summaries = {}
    for split in manifests:
        rows = [r for r in completed.values() if r['split'] == split]
        summaries[split] = {'samples': len(rows), 'over_cutoff': sum(r['exceeds_cutoff'] for r in rows),
            **{key: distribution([r[key] for r in rows]) for key in
               ['width', 'height', 'html_characters', 'target_text_tokens', 'multimodal_tokens']}}
    report = {'binding': binding, 'splits': summaries, 'lengths_sha256': sha256_file(checkpoint),
        'truncation_performed': False, 'weight_loading_performed': False,
        'notes': ['Native processor audit only; final LlamaFactory template counts still require verification.',
                  'No sample dropped, including overlong development/test targets.',
                  'Select training settings from training statistics; test content is not reviewed.']}
    write_json(args.output_dir/'summary.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
