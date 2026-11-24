# 表格图片 → HTML：Qwen3-VL-2B LoRA 与统一 TEDS 评测

把一个**已裁剪的英文科学文献表格**图片转成 HTML 表格。项目复用 Qwen3-VL-2B-Instruct 基座与 LlamaFactory/PEFT 训练 Q1 LoRA，使用 OmniDocBench 的 TEDS 指标，并提供 PaddleX/PP-OCR 传统流水线对照、完整失败保留和冻结验收证据。

- 基座：`Qwen/Qwen3-VL-2B-Instruct`，固定 revision `89644892e4d85e24eaac8bacfd4f463576704203`
- 适配器：Q1 LoRA（q/v，rank 8，alpha 16，约 6.4MB，随仓库提供）
- 演示：Gradio 单图上传、严格 HTML 校验、预览与下载
- 评测：TEDS / TEDS-S，失败全部保留在分母

## 最终结果

同一份新论文 test150（150 张 / 150 篇，项目未用论文留出）：

| 方法 | TEDS 0–100 | TEDS-S 0–100 | 有效 HTML | 截断 | 非法 HTML |
|---|---:|---:|---:|---:|---:|
| Q1 原设置（min_pixels 4096） | 71.62 | 80.53 | 135/150 | 15 | 0 |
| Q1 冻结新设置（min_pixels 65536） | **76.94** | 83.11 | 137/150 | 13 | 0 |
| PaddleX / PP-OCR（CPU） | 75.90 | **95.89** | 148/150 | 0 | 2 |

- 新设置相对原设置 +5.321 分，配对论文 bootstrap 95% CI [2.783, 7.993]；恢复 3 例、新增 1 例截断。
- 相对 Paddle 的平均差 +1.038 分，95% CI [-3.969, 5.907] **跨 0**，不能称稳定全面超过传统 OCR；Paddle 的结构完整性更好。
- TEDS 不是业务准确率；这是同源 PubTabNet val 的新论文留出测试，不是官方榜单，也不代表中文、手写、整页 PDF 或金融字段。
- **权重没有重训**：71.62→76.94 是固定 Q1 权重、只改推理视觉预算的干预结果，不能与训练增益相加。

完整证据：[最终验收](docs/FINAL_ACCEPTANCE.md)、[分析报告](evidence/final-test/analysis-report.md)、[统计](evidence/final-test/statistics.json)、[逐论文 CSV](evidence/final-test/paired-test150.csv)、[失败模式](docs/FAILURE_MODES.md)。

## 快速开始

### 0. 体检（不需要权重，也不会加载模型）

```bash
python -m table_vlm.doctor
```

逐项检查 Python、基座元数据/权重、Q1 adapter SHA256、输出目录、CUDA/transformers/peft/torchvision、端口。

### Windows（PowerShell 7）

```powershell
# 一次性准备：项目内 .venv + torch cu124 + 依赖 + 4.25GB 固定基座权重
.\scripts\setup_demo_windows.ps1

# 启动 UI
.\start_demo.ps1
# 浏览器打开 http://127.0.0.1:7861
```

### AutoDL / Linux GPU

```bash
bash scripts/setup_demo_autodl.sh   # .venv + torch cu124 + 4.25GB 权重 + doctor
bash scripts/run_demo_autodl.sh     # 监听 127.0.0.1:7861，并打印 SSH 隧道命令
```

```bash
ssh -L 7861:127.0.0.1:7861 -p <实例SSH端口> root@<实例地址>
```

### 单图 CLI

```bash
python -m table_vlm.demo --image examples/table.png
```

输出写到 `demo-output/requests/<时间戳>/`：`request.json`、`predictions.jsonl`、`raw-output.txt`、成功时的 `table.html`。截断或非法 HTML 只保留原文，不补标签包装成成功。

### 只跑离线测试（不需要 GPU / 权重）

```bash
python -m pip install -r requirements-data.txt pytest
python -m unittest discover -s tests -v
```

## 仓库里有什么

| 内容 | 是否包含 |
|---|---|
| `table_vlm/` 源码、`tests/`、`configs/`、`scripts/` | 有 |
| Q1 LoRA 权重与训练记录（约 6.4MB） | 有：`artifacts/q1-adapter/` |
| 基座 processor/tokenizer/config 元数据 | 有：`models/Qwen3-VL-2B-Instruct/` |
| 4.25GB 基座大权重 | **没有**，按固定 SHA256 下载 |
| 原始 PubTabNet 数据、训练标签、test 图片/预测 | **没有** |
| 虚拟环境、CUDA 驱动、账号 | **没有** |

下载器会逐文件核对 SHA256；`doctor` 会在启动前检查基座 revision 和 adapter 哈希。没有原始数据或封存预测时，不能宣称从零完整复现训练或离线重评分。

## 代码地图

| 路径 | 作用 |
|---|---|
| `table_vlm/demo.py` | Gradio UI、请求隔离、严格 HTML 校验与下载 |
| `table_vlm/demo_infer.py` | 冻结推理设置（min 65536 / max 196608、换行 prompt、BF16/SDPA、greedy、4096） |
| `table_vlm/doctor.py` | 环境/模型/适配器/CUDA/端口预检 |
| `table_vlm/paths.py` | 相对路径与 `TABLE_VLM_*` 环境变量 |
| `table_vlm/formal_data.py`、`formal_protocol.py` | 论文级数据准备与规则 |
| `table_vlm/evaluate.py`、`html_utils.py` | TEDS 评分适配与严格 HTML 检查 |
| `research/upstream/opendatalab__OmniDocBench/` | 固定版本的 TEDS 实现与版权来源 |
| `docs/` | 模型卡、数据政策、复现、最终验收 |
| `evidence/` | 冻结证据、逐论文分数、图表 |

## 评测与数据边界

- test150 清单 SHA256：`dbb0704a9675b04707210377c0f201a0449d9405650711322d93ffb233d6962c`；150 张/150 篇，已排除历史用过的 3469 篇论文与 5253 张图像/变体，并做字节/像素/感知哈希近重复检查。
- 全部 450 条（两组 VLM + Paddle）已在 CPU 独立逐条复算一致。
- 只做同源 PubTabNet val 留出；模型预训练是否见过这些公开论文未知，标注未做专家穷尽审核。
- 冻结设置：`min_pixels=65536`、`max_pixels=196608`、图片后换行 + `table_vlm.prompts.PROMPT`、BF16/SDPA、greedy、`repetition_penalty=1.0`、`max_new_tokens=4096`。
- 不要用 test150 继续调像素数、重复惩罚、prompt 或数据配方；新研究请另设开发集与验收集。

## 本机真实跑通记录

在 RTX 4060 Laptop 8GB 上用固定 revision 基座 + 包内 Q1 adapter 跑通：

- 单图：76.1 秒（生成 51.5 秒），峰值 allocated 约 4.13GiB；
- Gradio API：78.7 秒，预览/原始 HTML/下载均返回；
- 同一机器二次运行约 17.1 秒（缓存后观察，不是吞吐基准）。

只是单图功能记录，不替代冻结评测或证明整体速度。

## 许可与引用

本项目代码以 **Apache-2.0** 发布，见 [LICENSE](LICENSE)。上游组件与数据条款见 [NOTICE](NOTICE)、[docs/THIRD_PARTY_NOTICES.md](docs/THIRD_PARTY_NOTICES.md) 和 `research/upstream/`。

- 基座 Qwen3-VL-2B-Instruct：Apache-2.0
- Q1 adapter：基于 PubTabNet 标注训练；标注 CDLA-Permissive-1.0，图片受 PMC Open Access Subset 条款约束
- OmniDocBench TEDS、LlamaFactory、PaddleOCR/PaddleX、Gradio：Apache-2.0
- ImageHash：BSD-2-Clause

不要在简历或对外材料中写“自研基础模型/训练器/TEDS”“业务准确率 76.94%”“官方榜单成绩”或“全面超过 Paddle”；这些都不成立。
