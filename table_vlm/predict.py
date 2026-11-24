"""Single-image inference using Transformers' native Qwen3-VL implementation."""

import argparse
import json
import time
from pathlib import Path

from .io_utils import read_jsonl, sha256_file, unique_by_id, write_json
from .prompts import PROMPT


def main() -> None:
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True, help='Local directory acquired by download_model')
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--max-pixels', type=int, default=196608)
    parser.add_argument('--max-new-tokens', type=int, default=2048)
    parser.add_argument('--max-input-tokens', type=int, default=2048)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    receipt = json.loads((args.model/'source_receipt.json').read_text(encoding='utf-8'))
    rows = read_jsonl(args.manifest)
    unique_by_id(rows)
    if not rows or args.max_pixels < 4096 or args.max_new_tokens < 1:
        raise ValueError('Empty manifest or invalid generation limits')
    if not torch.cuda.is_available():
        raise RuntimeError('This baseline expects CUDA; use the data environment for CPU tasks')
    torch.manual_seed(42)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=False, local_files_only=True,
                                             min_pixels=4096, max_pixels=args.max_pixels)
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, trust_remote_code=False,
                local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda')
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter, local_files_only=True)
    model.eval()
    eos = model.generation_config.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count, statuses = 0, {}
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            image_path = (args.manifest.parent/row['image']).resolve()
            if not image_path.is_relative_to(args.manifest.parent.resolve()):
                raise ValueError('Image path escapes manifest directory')
            if sha256_file(image_path) != row['image_sha256']:
                raise ValueError(f'Image changed after data audit: {row["id"]}')
            started = time.perf_counter()
            generated_count = 0
            prediction, status, detail = '', 'ok', ''
            torch.cuda.reset_peak_memory_stats()
            inputs = generated = tokens = None
            try:
                with Image.open(image_path) as source_image:
                    image = source_image.convert('RGB')
                messages = [{'role': 'user', 'content': [
                    {'type': 'image', 'image': image}, {'type': 'text', 'text': PROMPT}]}]
                inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                       return_dict=True, return_tensors='pt').to('cuda')
                input_count = inputs['input_ids'].shape[-1]
                if input_count > args.max_input_tokens:
                    raise ValueError(f'Input length {input_count} exceeds configured limit; no silent truncation')
                torch.cuda.synchronize()
                started = time.perf_counter()
                with torch.inference_mode():
                    generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
                torch.cuda.synchronize()
                tokens = generated[0, input_count:]
                generated_count = len(tokens)
                prediction = processor.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                if generated_count >= args.max_new_tokens and int(tokens[-1]) not in eos_ids:
                    status = 'truncated'
            except torch.cuda.OutOfMemoryError as exc:
                status, detail = 'cuda_oom', str(exc).split('\n')[0]
            except (OSError, ValueError, RuntimeError) as exc:
                status, detail = 'generation_error', str(exc).split('\n')[0]
            record = {'id': row['id'], 'html': prediction, 'status': status, 'detail': detail,
                      'new_tokens': generated_count, 'generation_seconds': time.perf_counter()-started,
                      'peak_allocated_mib': torch.cuda.max_memory_allocated()/1024**2}
            stream.write(json.dumps(record, ensure_ascii=False)+'\n')
            stream.flush()
            count += 1
            statuses[status] = statuses.get(status, 0)+1
            print(json.dumps({'id': row['id'], 'status': status, 'tokens': generated_count,
                              'seconds': round(record['generation_seconds'], 2)}, ensure_ascii=False), flush=True)
            del inputs, generated, tokens
            torch.cuda.empty_cache()
    write_json(args.output.with_suffix('.run.json'), {
        'model': receipt['repo'], 'revision': receipt['revision'], 'adapter': str(args.adapter) if args.adapter else None,
        'manifest_sha256': sha256_file(args.manifest), 'prompt': PROMPT, 'samples': count, 'statuses': statuses,
        'max_pixels': args.max_pixels, 'max_new_tokens': args.max_new_tokens, 'do_sample': False,
        'torch': torch.__version__, 'device': torch.cuda.get_device_name(),
        'timing_note': 'Generation-only timing on successful samples; failed samples may include preprocessing.'})


if __name__ == '__main__':
    main()
