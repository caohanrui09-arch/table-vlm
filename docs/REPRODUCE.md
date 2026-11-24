# 复现与运行（2026-11-11）

> 2026-11-22 工程化改进：演示已改为项目相对路径和单环境优先；下面“本机演示”已更新。其余 `<DATA_ROOT>/...` 是原作者机器的证据位置，仅用于审计，不是接收电脑必须复制的路径。

当前交付是原 Q1 权重加独立测试前冻结的最低视觉预算 65536。区分演示功能、离线重评分与历史训练重建；日常展示无需租卡或重跑测试。

## 固定环境与模型

- 云实验：4090，Torch2.6.0+cu124、Transformers4.57.1、PEFT0.18.1；训练LlamaFactory0.9.6.dev0固定提交c683e59edd8f893abc5e1403184a1d0644817d2a。
- 本机UI：项目.venv，Gradio6.31.0/gradio_client2.7.2；GPU用父目录.venv及项目.inference-deps，Torch2.6.0+cu124/Transformers4.57.1/PEFT0.21.2。本机功能验证不代替原云benchmark。
- Paddle CPU另环境：paddlepaddle3.2.2、paddleocr3.3.2、paddlex3.3.13；不作CPU/GPU速度排名。
- 基座revision：89644892e4d85e24eaac8bacfd4f463576704203；Q1 adapter SHA：5ba4f106312a738283916fb5c691376584539d0824c597e02f6485c995da250c。

沿用已锁环境，不要求升级最新版。依赖锁/来源在requirements、DATA_POLICY、THIRD_PARTY_NOTICES；未声称在全新空机器完成一键安装验收。

## 本机演示

2026-11-22 改进版把路径改为项目相对路径。首次准备：

```powershell
.\scripts\setup_demo_windows.ps1
python -m table_vlm.doctor
```

用 PowerShell 7 在项目目录运行：

```powershell
.\start_demo.ps1
```

打开 http://127.0.0.1:7861，上传裁剪英文表格，或使用明确标注的训练集功能样例 `examples/table.png`；查看原始 HTML、预览与下载。`stop_demo.ps1` 只结束 PID/启动时间匹配的本服务；`start_demo.ps1` 识别并安全更新旧版服务，不盲杀端口进程。

单图功能运行（输出写到 `demo-output/requests/`）：

```powershell
python -m table_vlm.demo --image .\examples\table.png
```

演示调用 `table_vlm/demo_infer.py`，复用原生 Transformers 路径；旧 predict/A4/A5 实验不随公开仓库提供；当前入口只使用 table_vlm/demo_infer.py。固定 min65536/max196608、图片后换行 prompt、BF16/SDPA、greedy、repetition_penalty 1、输出 4096。请求回执记录实际网格、模型/图片摘要和版本；UI 不提供调参滑块。

迁移可显式给 demo 传 `--model/--adapter/--gpu-python/--output-root`，或设置 `TABLE_VLM_MODEL` 等环境变量；权重必须仍为冻结 Q1 和基座 revision。UI 与 GPU 环境可分开，见 LOCAL_DEMO.md。如果 worker 的 `torchvision` 等依赖在单独目录，加 `--extra-pythonpath` 或 `TABLE_VLM_PYTHONPATH`。

本机 RTX 4060 8GB 已用原 Q1 和用户下载的固定 revision 基座跑通单图：总耗时约 76 秒（生成约 51.5 秒），峰值 allocated 约 4.13GiB；Gradio API 约 78.7 秒。仅功能记录，不是吞吐基准。

## 数据与证据

源码在本项目；所有新增请求、缓存、下载、日志和结果在<DATA_ROOT>/doc-table-vlm。基座沿用已有models目录，不重复复制4.25GB模型；完整数据不打进源码包。

- 原训练：prepared/formal-4032-clean-v1，2990条；configs/lora_24gb.yaml，两epoch/748步，不为演示重训。
- A4开发：a4-resolution-20261106，dev150用于此前选择，不当新测试。
- A5新测试：a5-final-resolution-20261111/test-data/test.jsonl，SHA dbb0704a9675b04707210377c0f201a0449d9405650711322d93ffb233d6962c。
- 测试前记录：同目录protocol-before-test.json、historical-exclusion.json、test-freeze.json和final-test-freeze.json，全部保留不改。
- 两VLM预测：received-final-resolution-20261111T002744Z/runs/final-resolution-20261111T002744Z；Paddle分片/合并在同根paddle。
- 独立评分：analysis-final-resolution-20261111T002744Z；报告与逐论文CSV在其analysis-output。
- 演示：demo/requests与demo/examples；来源标注为train，历史例图备份保留。

## 离线重评分：只读已有预测

450条已经独立复算一致。需要审计时只重新评分已有原始输出，不调用模型、筛样或改规则；选新的空输出目录：

```powershell
$py = '<DATA_ROOT>/doc-table-vlm/a2-20261104/cpu-venv/Scripts/python.exe'
& $py -m table_vlm.evaluate `
  --manifest <DATA_ROOT>/doc-table-vlm/a5-final-resolution-20261111/test-data/test.jsonl `
  --predictions <DATA_ROOT>/doc-table-vlm/a5-final-resolution-20261111/received-final-resolution-20261111T002744Z/runs/final-resolution-20261111T002744Z/min65536.jsonl `
  --output <DATA_ROOT>/doc-table-vlm/offline-review-NEW-EMPTY-DIRECTORY
```

演示检查：同CPU环境执行python -m unittest discover -s tests -p test_demo.py -v。真实GPU/API功能回执在delivery-20261111，只使用train样例，不算新质量成绩。

## 历史重建边界

历史 A2–A5 实验不随公开仓库提供；保留的是冻结协议、汇总证据和最终验收，不重复启动已完成测试。当前test已用于验收，不再用来选模型、调像素预算、补标签或挑成功图。

V1训练增益、A4开发分数、A5独立测试干预增益不能相加。旧报告/原始结果不替换；最新结论见FINAL_ACCEPTANCE.md。新研究需另设开发与验收数据。
