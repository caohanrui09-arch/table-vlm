"""Check real multimodal sequence lengths without loading model weights."""

import argparse
import json
from pathlib import Path

from .io_utils import read_jsonl, sha256_file, write_json
from .prompts import PROMPT


def main() -> None:
    from PIL import Image
    from transformers import AutoProcessor

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-pixels', type=int, default=196608)
    parser.add_argument('--cutoff-len', type=int, default=4096)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True, trust_remote_code=False,
                                             min_pixels=4096, max_pixels=args.max_pixels)
    rows = []
    for row in read_jsonl(args.manifest):
        image_path = (args.manifest.parent/row['image']).resolve()
        if not image_path.is_relative_to(args.manifest.parent.resolve()):
            raise ValueError('Image escapes manifest directory')
        if sha256_file(image_path) != row['image_sha256']:
            raise ValueError(f'Image changed after data audit: {row["id"]}')
        with Image.open(image_path) as source_image:
            image = source_image.convert('RGB')
        answer = row['html'].replace('<html><body>', '').replace('</body></html>', '')
        user = {'role': 'user', 'content': [{'type': 'image', 'image': image}, {'type': 'text', 'text': PROMPT}]}
        encoded = processor.apply_chat_template([user, {'role': 'assistant', 'content': [{'type': 'text', 'text': answer}]}],
                        tokenize=True, return_dict=True, return_tensors='pt', add_generation_prompt=False)
        length = encoded['input_ids'].shape[-1]
        rows.append({'id': row['id'], 'multimodal_tokens': length, 'exceeds_cutoff': length > args.cutoff_len,
                     'image_grid_thw': encoded['image_grid_thw'].tolist()})
    report = {'samples': len(rows), 'manifest_sha256': sha256_file(args.manifest), 'max_pixels': args.max_pixels,
              'cutoff_len': args.cutoff_len, 'too_long': sum(r['exceeds_cutoff'] for r in rows),
              'max_tokens': max((r['multimodal_tokens'] for r in rows), default=0), 'rows': rows,
              'note': 'Native processor length check; verify the final LlamaFactory template/preprocessing count before full training. No truncation or dropping is performed.'}
    write_json(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}))
    if report['too_long']:
        raise SystemExit('Overlong targets: reduce table complexity or explicitly revise the protocol before training')


if __name__ == '__main__':
    main()
