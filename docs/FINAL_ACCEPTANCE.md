# 新论文独立测试：最终验收（2026-11-11）

收尾更新：冻结设置已接入演示，现已更新并用非测试train样例通过真实API/预览/下载验证。测试分数和冻结协议未改变；演示入口见 [LOCAL_DEMO.md](LOCAL_DEMO.md)，项目入口见 [../README.md](../README.md)。下文“尚未自动修改”指本次验收时状态。

冻结设置通过本次独立测试，项目最终VLM方案保留原Q1权重，最低视觉预算65536、最高196608，其余生成/评分协议不变。未依据test另搜设置，展示服务未自动修改。

| 同一新论文test150 | Q1原设置 | Q1改进设置 | Paddle CPU |
|---|---:|---:|---:|
| TEDS（0–100） | 71.62 | **76.94** | 75.90 |
| TEDS-S（0–100） | 80.53 | 83.11 | **95.89** |
| 有效HTML | 135 | 137 | 148 |
| 截断 | 15 | 13 | 0 |
| 非法HTML | 0 | 0 | 2 |

主比较改进-原设置提高5.321分，配对论文bootstrap95%区间[2.783,7.993]分；逐论文胜/平/负63/78/9。恢复3例、新增1例截断，仍有12例共同截断；没有修复或丢弃失败。与Paddle的平均差+1.038分，95%区间[-3.969,5.907]跨0，不能称全面稳定超过传统OCR。不同技术路线的结构完整性代价必须报告。

验收证据：结果ZIP539892字节、24文件CRC通过，SHA6d9d1a839e7ba1ccae1ceb3be42876cd1580643d1550eaa8d596807157edc4c6。全部450条CPU重评分逐条及汇总完全一致；Paddle四分片/合并/模型配置和冻结授权通过；74张未变输入对应输出均相同；5项关键计分/恢复测试通过。两图PNG已查看、PDF同时提供。

- [完整分析报告](<DATA_ROOT>/doc-table-vlm/a5-final-resolution-20261111/analysis-final-resolution-20261111T002744Z/analysis-output/analysis-report.md)
- [逐论文三法CSV](<DATA_ROOT>/doc-table-vlm/a5-final-resolution-20261111/analysis-final-resolution-20261111T002744Z/analysis-output/paired-test150.csv)
- [新增与恢复失败案例](<DATA_ROOT>/doc-table-vlm/a5-final-resolution-20261111/analysis-final-resolution-20261111T002744Z/analysis-output/case-review.md)
- [验收归档](<DATA_ROOT>/doc-table-vlm/a5-final-resolution-20261111/dist/final-resolution-20261111T002744Z-local-analysis.zip)

可用于简历的准确表述：

> 基于Qwen3-VL-2B与LlamaFactory完成表格图像到HTML项目，构建论文级隔离和统一TEDS评测；通过重复生成与视觉输入诊断，固定LoRA权重提高最低视觉预算，在预先冻结的150篇新论文测试上将TEDS从71.62提高至76.94，截断从15降至13，并保留Paddle对照、失败样例和历史负结果。

不能把该提升与V1另一测试集的训练增益相加，也不能写成自研模型、业务准确率76.94%、循环已完全解决、多seed稳定或跨域验证。只是同源PubTabNet val的新论文留出、固定权重一次成对推理；预训练污染和标注误差未知。

下一步收尾演示/复现说明与面试材料，不再追加租卡实验追本次test分数。若研究新方向，需另设开发与验收数据。累计预算上限100元、最新用户报告50元保留为历史账单信息，不推算当前扣费；本机验收没有新增租卡费用。
