"""Local single-table demo: Gradio UI + the frozen, isolated inference CLI.

Paths are resolved relative to the project (see :mod:`table_vlm.paths`) and can
be overridden by CLI flags or ``TABLE_VLM_*`` environment variables.  The model
worker can live in the same environment as the UI, which is the normal setup
for a fresh machine; a separate CUDA environment is still supported with
``--gpu-python``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
import uuid

from PIL import Image

from . import paths
from .html_utils import canonical_table, InvalidTable, NORMALIZATION_VERSION
from .io_utils import read_jsonl, sha256_file, write_json, write_jsonl


ROOT = paths.PROJECT_ROOT
DEMO_REVISION = 'a5-min65536-v1'
MIN_PIXELS = 65536
MAX_PIXELS = 196608
MAX_NEW_TOKENS = 4096


def export_document(fragment: str) -> str:
    table = canonical_table(fragment).removeprefix('<html><body>').removesuffix('</body></html>')
    return ('<!doctype html><html><head><meta charset="utf-8">'
            '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; '
            'style-src \'unsafe-inline\'; base-uri \'none\'; form-action \'none\'">'
            '<title>识别表格</title><style>body{font-family:system-ui;margin:16px}'
            'table{border-collapse:collapse}td,th{border:1px solid #ccd3dd;padding:7px;'
            'vertical-align:top}thead{background:#f0f3f7}</style></head><body>'+table+'</body></html>')


def present_prediction(prediction: dict, output: Path) -> dict:
    """Only completed, strictly valid HTML is rendered or exported as HTML."""
    raw = prediction.get('html', '')
    (output/'raw-output.txt').write_text(raw, encoding='utf-8')
    result = {'generation_status': prediction.get('status'), 'html_valid': False,
              'html_file': None, 'preview': '', 'raw': raw,
              'normalization_version': NORMALIZATION_VERSION}
    if prediction.get('status') != 'ok':
        result['message'] = ('输出达到长度上限，原始内容已保留。' if prediction.get('status') == 'truncated'
                             else '推理未完成，错误信息已保留在本次记录中。')
        return result
    try:
        document = export_document(raw)
    except (InvalidTable, ValueError, TypeError) as error:
        result.update(message='HTML 格式未通过检查，原始内容已保留。', error_detail=str(error))
        return result
    destination = output/'table.html'
    destination.write_text(document, encoding='utf-8')
    result.update(html_valid=True, html_file=str(destination), message='识别完成，请核对文字、数字及合并单元格。',
                  preview='<iframe title="表格预览" sandbox="" style="width:100%;height:420px;border:0" '
                          'srcdoc="'+html.escape(document, quote=True)+'"></iframe>')
    return result


class DemoService:
    def __init__(self, output: Path | None = None, model: Path | None = None,
                 adapter: Path | None = None, python: Path | None = None,
                 label: str = 'Qwen3-VL-2B 表格识别', timeout: float = 900,
                 device: str = 'auto', adapter_sha: str = paths.Q1_SHA256,
                 base_revision: str = paths.BASE_REVISION,
                 example: Path | None = None,
                 extra_pythonpath: str | Path | None = None) -> None:
        self.output = paths.output_path(output)
        self.model = paths.model_path(model)
        self.adapter = paths.adapter_path(adapter)
        self.python = paths.gpu_python(python)
        self.label = label
        self.timeout = timeout
        self.device = device
        self.adapter_sha = adapter_sha
        self.base_revision = base_revision
        self.example = paths.example_path(example)
        self.extra_pythonpath = paths.extra_pythonpath(extra_pythonpath)
        self.lock = threading.Lock()
        self._validated = False

    def validate(self) -> None:
        """Fail early, with an actionable message, before spending GPU time."""
        if self._validated:
            return
        for item in [self.python, self.model/'config.json', self.model/'preprocessor_config.json',
                     self.model/'tokenizer_config.json', self.model/'tokenizer.json',
                     self.adapter/'adapter_model.safetensors', self.adapter/'adapter_config.json']:
            if not item.is_file():
                raise FileNotFoundError(f'演示依赖缺失: {item}')
        weight = self.model/paths.MODEL_WEIGHT
        if not weight.is_file():
            raise FileNotFoundError(
                f'基座权重缺失: {weight}\n先运行: python -m table_vlm.download_model '
                f'--metadata-manifest research/qwen2b-fixed-manifest.json --transport modelscope')
        receipt = self.model/'source_receipt.json'
        if not receipt.is_file():
            raise FileNotFoundError(
                f'模型回执缺失: {receipt}\n请用 table_vlm.download_model 下载，或运行 table-vlm-doctor 查看缺什么。')
        revision = json.loads(receipt.read_text(encoding='utf-8')).get('revision')
        if revision != self.base_revision:
            raise ValueError(f'基座 revision 与冻结版本不一致: {revision!r} != {self.base_revision!r}')
        if sha256_file(self.adapter/'adapter_model.safetensors') != self.adapter_sha:
            raise ValueError('演示适配器与已冻结 Q1 版本不一致，请核对 --adapter 路径。')
        self._validated = True

    def inference_command(self, manifest: Path, predictions: Path) -> list[str]:
        return [str(self.python), '-m', 'table_vlm.demo_infer', '--model', str(self.model),
                '--adapter', str(self.adapter), '--manifest', str(manifest), '--output', str(predictions),
                '--min-pixels', str(MIN_PIXELS), '--max-pixels', str(MAX_PIXELS),
                '--max-new-tokens', str(MAX_NEW_TOKENS), '--device', self.device]

    def _run_inference(self, command: list[str], folder: Path, env: dict[str, str]) -> int:
        with (folder/'inference.log').open('w', encoding='utf-8') as log:
            completed = subprocess.run(
                command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                timeout=self.timeout,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        return completed.returncode

    def run(self, image_path: str | os.PathLike[str]) -> dict:
        if not image_path:
            raise ValueError('请先上传一张裁剪好的表格图片。')
        image_path = Path(image_path)
        if not image_path.is_file() or image_path.stat().st_size > 20*1024**2:
            raise ValueError('请选择不超过 20MB 的图片文件。')
        with Image.open(image_path) as im:
            if im.format not in {'PNG','JPEG','WEBP'} or getattr(im, 'n_frames', 1) != 1:
                raise ValueError('请选择单张 PNG、JPEG 或 WebP 图片。')
            if im.width*im.height > 30_000_000:
                raise ValueError('图片过大，请先裁剪表格区域。')
            extension = {'PNG':'.png','JPEG':'.jpg','WEBP':'.webp'}[im.format]
            im.verify()
        self.validate()
        with self.lock:
            started = time.perf_counter()
            name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
            folder = self.output/'requests'/name
            folder.mkdir(parents=True, exist_ok=False)
            image = folder/f'input{extension}'
            shutil.copyfile(image_path, image)
            image_digest = sha256_file(image)
            sample_id = 'upload-'+image_digest[:16]
            write_jsonl(folder/'input.jsonl', [{'id':sample_id, 'image':image.name, 'image_sha256':image_digest}])
            write_json(folder/'request.json', {'input_sha256':image_digest,
                'model_receipt_sha256':sha256_file(self.model/'source_receipt.json'),
                'adapter_sha256':sha256_file(self.adapter/'adapter_model.safetensors'),
                'adapter':str(self.adapter), 'stage':'interactive single-image inference',
                'demo_revision':DEMO_REVISION, 'min_pixels':MIN_PIXELS,
                'max_pixels':MAX_PIXELS, 'max_new_tokens':MAX_NEW_TOKENS,
                'prompt_image_separator':'newline', 'do_sample':False, 'repetition_penalty':1.0,
                'device':self.device, 'quality_benchmark':False})
            worker_path = [str(ROOT), *map(str, self.extra_pythonpath)]
            if os.environ.get('PYTHONPATH'):
                worker_path.append(os.environ['PYTHONPATH'])
            env = {**os.environ, 'HF_HUB_OFFLINE':'1', 'TRANSFORMERS_OFFLINE':'1',
                   'PYTHONUNBUFFERED':'1', 'CUDA_VISIBLE_DEVICES':'0',
                   'PYTHONPATH':os.pathsep.join(worker_path)}
            command = self.inference_command(folder/'input.jsonl', folder/'predictions.jsonl')
            try:
                returncode = self._run_inference(command, folder, env)
                if returncode != 0:
                    raise RuntimeError('推理进程未完成，详细日志已保存到 inference.log。')
                predictions = read_jsonl(folder/'predictions.jsonl')
                if len(predictions)!=1 or predictions[0].get('id')!=sample_id:
                    raise RuntimeError('推理结果与输入不一致。')
                result = present_prediction(predictions[0],folder)
                result.update(seconds=time.perf_counter()-started, output_directory=str(folder),
                              prediction=predictions[0])
                write_json(folder/'result.json', {k:v for k,v in result.items() if k not in {'preview','raw'}})
                return result
            except (RuntimeError, subprocess.TimeoutExpired, OSError, ValueError) as error:
                write_json(folder/'failure.json', {'error':str(error),'seconds':time.perf_counter()-started})
                raise


def create_app(service: DemoService):
    os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
    os.environ['GRADIO_TEMP_DIR'] = str(service.output/'ui-cache')
    import gradio as gr

    def recognize(path):
        try:
            result = service.run(path)
        except (ValueError,OSError,RuntimeError,subprocess.TimeoutExpired) as error:
            raise gr.Error(str(error)) from error
        return (f"{result['message']}  总耗时 {result['seconds']:.1f} 秒。",
                result['preview'],result['raw'],result['html_file'])

    with gr.Blocks(title='表格图片识别',analytics_enabled=False) as app:
        gr.Markdown('# 表格图片 → HTML\n上传裁剪好的英文表格图片，查看并下载识别结果。当前模型：'+html.escape(service.label))
        with gr.Row():
            with gr.Column():
                image = gr.Image(type='filepath',format='png',sources=['upload'],label='表格图片')
                button = gr.Button('识别表格',variant='primary')
                status = gr.Markdown('等待上传图片。')
                if service.example.is_file():
                    gr.Examples(examples=[[str(service.example)]],inputs=[image],cache_examples=False,
                                label='训练集功能样例（不代表测试效果）')
            with gr.Column():
                preview = gr.HTML(label='表格预览')
                download = gr.File(label='下载 HTML',interactive=False)
        with gr.Accordion('查看原始 HTML',open=False):
            source = gr.Code(language='html',label='模型原始输出',interactive=False)
        button.click(recognize,inputs=[image],outputs=[status,preview,source,download],
                     api_name='recognize',concurrency_limit=1)
    return app.queue(default_concurrency_limit=1,max_size=4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--image',type=Path,help='Run a single CLI image instead of the UI')
    parser.add_argument('--host',default='127.0.0.1',help='Bind address; keep localhost unless you add a tunnel')
    parser.add_argument('--port',type=int,default=7861)
    parser.add_argument('--model',type=Path,help=f'base model directory (env {paths.ENV_MODEL})')
    parser.add_argument('--adapter',type=Path,help=f'Q1 adapter directory (env {paths.ENV_ADAPTER})')
    parser.add_argument('--output-root',type=Path,help=f'demo output directory (env {paths.ENV_OUTPUT})')
    parser.add_argument('--gpu-python',type=Path,help=f'model worker interpreter (env {paths.ENV_GPU_PYTHON})')
    parser.add_argument('--model-label',default='Qwen3-VL-2B 表格识别')
    parser.add_argument('--device',choices=['auto','cuda','cpu'],default='auto',
                        help='worker device; auto prefers CUDA and falls back to CPU')
    parser.add_argument('--timeout',type=float,default=900,help='per-image worker timeout in seconds')
    parser.add_argument('--example',type=Path,help='override the bundled train example image')
    parser.add_argument('--extra-pythonpath',help='worker-only package directories (os.pathsep-separated); '
                        f'also settable with {paths.ENV_EXTRA_PYTHONPATH}')
    parser.add_argument('--skip-preflight',action='store_true',help='skip the model/adapter check at startup')
    args = parser.parse_args()
    service = DemoService(output=args.output_root,model=args.model,adapter=args.adapter,
                          python=args.gpu_python,label=args.model_label,timeout=args.timeout,
                          device=args.device,example=args.example,
                          extra_pythonpath=args.extra_pythonpath)
    if not args.skip_preflight:
        try:
            service.validate()
        except (OSError,ValueError,KeyError,json.JSONDecodeError) as error:
            parser.error(f'预检失败，先运行 `python -m table_vlm.doctor`\n{error}')
    if args.image:
        result = service.run(str(args.image))
        print(json.dumps({k:v for k,v in result.items() if k not in {'preview','raw'}},ensure_ascii=False))
        return
    app = create_app(service)
    app.launch(server_name=args.host,server_port=args.port,share=False,inbrowser=False,
               allowed_paths=[str(service.output)],blocked_paths=[str(service.model),str(service.adapter)],
               max_file_size='20mb',enable_monitoring=False,footer_links=[],run_history=False,
               theme='default',ssr_mode=False,mcp_server=False)


if __name__ == '__main__':
    main()
