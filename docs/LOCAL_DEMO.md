# 本机单表演示（已验收设置）

当前使用同一 Qwen3-VL-2B + Q1 LoRA，接入测试前冻结且独立验收通过的 min65536/max196608、图片后换行 prompt。greedy、repetition_penalty 1、BF16/SDPA、输出 4096。模型和图片均在本机处理。

> 2026-11-22 工程化改进：路径改为项目相对路径，默认单环境运行，新增 `demo-output/requests/` 输出和 `table-vlm-doctor` 预检。原 `<DATA_ROOT>/...` 请求目录只是原作者机器的历史证据。

## 使用

```powershell
# 一次性准备（项目内 .venv、依赖、4.25GB 固定权重）
.\scripts\setup_demo_windows.ps1

# 启动；浏览器打开 http://127.0.0.1:7861
.\start_demo.ps1
```

上传裁剪英文表格，或点击包内训练集功能例图 `examples/table.png`（PMC3485118_004_00，非测试图），查看原始 HTML、严格校验后的预览和 HTML 下载。请求与输出写到 `demo-output/requests/<时间戳>/`。

`stop_demo.ps1` 只停止 PID/启动时间匹配的本服务。`start_demo.ps1` 发现同端口已有本 demo 时不会重复启动；用 `-Force` 才会先停旧服务。

没有独立 GPU 环境时，模型 worker 默认与 UI 共用同一解释器；如果仍想分开：

```powershell
.\start_demo.ps1 -GpuPython D:\envs\table-vlm-gpu\Scripts\python.exe
```

如果 worker 的 `torchvision` 等包装在单独目录（原机器的 `.inference-deps` 结构），再传 `-ExtraPythonPath` 或设置 `TABLE_VLM_PYTHONPATH`：

```powershell
.\start_demo.ps1 -GpuPython D:\envs\table-vlm-gpu\Scripts\python.exe -ExtraPythonPath D:\envs\table-vlm-extra
```

不想用 PowerShell 7 时可用 `python scripts\run_demo.py --gpu-python ... --extra-pythonpath ...`。

本机 RTX 4060 Laptop 8GB 实测：单图总耗时约 76 秒（生成约 51.5 秒），峰值 allocated 约 4.13GiB；Gradio API 约 78.7 秒。只是单图功能记录，不是吞吐基准。

## 输出与分数

截断/非法/推理错误保留原始记录，不补标签、不伪装成功，不提供成功 HTML 下载。有效 HTML 仍可能有错字、数字或合并错误，需要核对；预览使用无脚本 sandbox iframe，下载带 CSP。

正式新论文 test150 成绩来自云 4090/PEFT 0.18.1：原 TEDS 71.62→新 76.94，截断 15→13；Paddle 75.90 且结构完整性更好。本机 4060 8GB 只验证交互功能，不替代冻结评分或证明吞吐性能。

## 技术与回执

UI 和 GPU worker 可同环境，也可用 `--gpu-python` 分开。worker `table_vlm/demo_infer.py` 支持 `--device auto|cuda|cpu`；GPU 上仍是冻结设置，CPU 只用于链路冒烟，速度很慢。`request.json`、`predictions.run.json` 保留像素预算、prompt、版本、设备、模型/输入摘要和真实网格。

真实 API/HTML 下载验证见原项目 `delivery-20261111/qa-api.json`（未随包，train 图，不生成 test 预测）。冷启动加载较慢，单样例耗时不是总体基准。本版未验证整页 PDF、中文财报、手写或 Agent 任务。
