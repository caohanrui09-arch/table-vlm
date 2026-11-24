# 数据来源与本地研究使用记录

核验日期：2026-10-29（Asia/Hong_Kong）。本项目保留数据来源及声明，不给原始图片自行换许可证。

## 已取得的 20 个官方样例

- 来源：[IBM PubTabNet 官方 GitHub](https://github.com/ibm-aur-nlp/PubTabNet)，固定提交 `8ffde9024bd331f5a61c3c549fdf30dd3c3e43a5`。
- [数据 LICENSE.md](https://github.com/ibm-aur-nlp/PubTabNet/blob/8ffde9024bd331f5a61c3c549fdf30dd3c3e43a5/LICENSE.md)规定标注为 CDLA-Permissive-1.0；图片遵循 PMC Open Access Subset 使用条款，不能用标注许可证替代图片许可。
- 24 个来源文件均记录 Git blob SHA1、SHA256 和字节数；原始数据未修改，增强数据另存并记录操作。
- 原 split 全部为 train；14/3/3 本地划分仅用于工程 smoke。它不是官方测试集，也不能推断总体识别率。

## 全量入口的标签差异与处理

上游 README 指向的 [ajimeno/PubTabNet](https://huggingface.co/datasets/ajimeno/PubTabNet)为约 11.2GB 压缩包，数据卡标记 CDLA-Sharing-1.0，与 GitHub 标注许可说明不同。这两项声明均保留，不能凭推测将一个改成另一个。

重新核对 [CDLA-Permissive-1.0](https://cdla.dev/permissive-1-0/) 和 [CDLA-Sharing-1.0](https://cdla.dev/sharing-1-0/) 后，两者第 2 节都许可使用和计算分析，区别主要涉及发布数据的条件。本项目可以继续按保留来源和声明的方式做本地研究准备；**单纯的元数据标签差异不再作为停止本地准备的理由**。先前据此暂停整个数据准备的处理过于保守。

项目实现方式：

1. 数据、增强图片和完整训练标注不进入公共 Git；提供来源、下载和重建步骤。
2. 下载全量包时固定上游 revision 和文件摘要，保留收到的说明文件；不声称已下载或核对尚未取得的包内内容。
3. 本地计算分析、模型实验与日后发布数据分开处理；若要公开数据或数据子集，再核对其收到版本的条款、变更说明、署名和图片来源要求。
4. 2026-10-30完整包已下载到D盘并两次核对固定SHA256，正式子集正在构建。不能把20例复制或增强到3000条后称为3000条独立真实样本；实际样本数以正式清单和审计为准。

完整包内实际附带 `LICENSE.txt`，标题为 **Community Data License Agreement – Permissive – Version 1.0**，10462字节，SHA256为 `52aa39b7c72a3712b1e1ab0b2d6b9d790f443f985407672b9df8edab9eb64712`。原文与提取回执保存在 `<DATA_ROOT>/doc-table-vlm/raw/archive-notices`。这份包内声明与原GitHub一致；HF数据卡Sharing标签仍作为来源差异记录，不由本项目修改。图片继续保留原项目对PMC OA使用条款的说明。

## 运行与对照边界

CPU 的传统表格基线复用 PaddleX 原生 `table_recognition` 流水线及官方 PP-OCR/SLANet_plus 模型。代码许可为 Apache-2.0；模型下载地址和实际内容哈希单独留存。它是轻量 OCR 流水线，不是 PaddleOCR-VL，也不是默认的 PP-TableMagic V2 配置。

本地缓存 Qwen3-VL-2B 已取得完整权重，固定 revision 为 `89644892e4d85e24eaac8bacfd4f463576704203`，4,255,140,312 字节的 safetensors 已匹配官方 LFS SHA256。权重下载完成不代表 GPU 推理或训练已经完成。
