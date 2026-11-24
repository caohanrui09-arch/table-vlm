"""Demo-only adapter of the original predict CLI; frozen A5 preprocessing.

The generated artifacts must stay identical to the frozen setting when a CUDA
device is used: min_pixels 65536, max_pixels 196608, newline before the prompt,
BF16/SDPA, greedy decoding, repetition_penalty 1.0, max_new_tokens 4096.

``--device auto`` keeps that behaviour on a GPU host and allows a slow CPU
smoke test on a machine without CUDA.  CPU output is for plumbing checks only,
not a replacement for the sealed GPU benchmark.
"""

import argparse
import importlib.metadata
import json
import time
from pathlib import Path

from .io_utils import read_jsonl, sha256_file, unique_by_id, write_json
from .prompts import PROMPT


def _select_device(requested: str, torch) -> str:
    if requested == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but torch.cuda.is_available() is false; use --device cpu or a GPU host')
    return requested


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', type=Path, required=True, help='Local directory acquired by download_model')
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--min-pixels', type=int, default=65536)
    parser.add_argument('--max-pixels', type=int, default=196608)
    parser.add_argument('--max-new-tokens', type=int, default=4096)
    parser.add_argument('--max-input-tokens', type=int, default=2048)
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto',
                        help='auto = CUDA when available, otherwise CPU (slow smoke test)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if args.output.exists():
        raise FileExistsError(args.output)
    device = _select_device(args.device, torch)
    receipt = json.loads((args.model/'source_receipt.json').read_text(encoding='utf-8'))
    rows = read_jsonl(args.manifest)
    unique_by_id(rows)
    if not rows or args.max_pixels < 4096 or args.max_new_tokens < 1:
        raise ValueError('Empty manifest or invalid generation limits')
    torch.manual_seed(args.seed)
    dtype = torch.bfloat16 if device == 'cuda' else torch.float32
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=False, local_files_only=True,
                                             min_pixels=args.min_pixels, max_pixels=args.max_pixels)
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, trust_remote_code=False,
                local_files_only=True, dtype=dtype, attn_implementation='sdpa').to(device)
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
            if device == 'cuda':
                torch.cuda.reset_peak_memory_stats()
            inputs = generated = tokens = None
            try:
                with Image.open(image_path) as source_image:
                    image = source_image.convert('RGB')
                messages = [{'role': 'user', 'content': [
                    {'type': 'image', 'image': image}, {'type': 'text', 'text': '\n'+PROMPT}]}]
                inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                       return_dict=True, return_tensors='pt').to(device)
                input_count = inputs['input_ids'].shape[-1]
                if input_count > args.max_input_tokens:
                    raise ValueError(f'Input length {input_count} exceeds configured limit; no silent truncation')
                if device == 'cuda':
                    torch.cuda.synchronize()
                started = time.perf_counter()
                with torch.inference_mode():
                    generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, repetition_penalty=1.0)
                if device == 'cuda':
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
                      'peak_allocated_mib': (torch.cuda.max_memory_allocated()/1024**2) if device == 'cuda' else 0,
                      'image_grid_thw': inputs['image_grid_thw'][0].tolist() if inputs is not None else None,
                      'input_tokens': inputs['input_ids'].shape[-1] if inputs is not None else None,
                      'device': device}
            stream.write(json.dumps(record, ensure_ascii=False)+'\n')
            stream.flush()
            count += 1
            statuses[status] = statuses.get(status, 0)+1
            print(json.dumps({'id': row['id'], 'status': status, 'tokens': generated_count,
                              'seconds': round(record['generation_seconds'], 2)}, ensure_ascii=False), flush=True)
            del inputs, generated, tokens
            if device == 'cuda':
                torch.cuda.empty_cache()
    write_json(args.output.with_suffix('.run.json'), {
        'model': receipt['repo'], 'revision': receipt['revision'], 'adapter': str(args.adapter) if args.adapter else None,
        'manifest_sha256': sha256_file(args.manifest), 'prompt': '\n'+PROMPT, 'samples': count, 'statuses': statuses,
        'min_pixels': args.min_pixels, 'max_pixels': args.max_pixels, 'repetition_penalty': 1.0,
        'versions': {k:importlib.metadata.version(k) for k in ['torch','transformers','peft']},
        'adapter_sha256':sha256_file(args.adapter/'adapter_model.safetensors') if args.adapter else None,
        'max_new_tokens': args.max_new_tokens, 'do_sample': False,
        'torch': torch.__version__, 'device': device,
        'cuda_device': torch.cuda.get_device_name() if device == 'cuda' else None,
        'timing_note': 'Generation-only timing on successful samples; failed samples may include preprocessing.'})


if __name__ == '__main__':
    main()
