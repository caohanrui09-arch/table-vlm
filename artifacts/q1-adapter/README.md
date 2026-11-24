---
library_name: peft
license: other
base_model: models/Qwen3-VL-2B-Instruct
tags:
- base_model:adapter:models/Qwen3-VL-2B-Instruct
- llama-factory
- lora
- transformers
pipeline_tag: text-generation
model-index:
- name: q1-adapter
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# q1-adapter

This model is a fine-tuned version of [models/Qwen3-VL-2B-Instruct](https://huggingface.co/models/Qwen3-VL-2B-Instruct) on the table_train dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0928

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 1
- eval_batch_size: 1
- seed: 42
- gradient_accumulation_steps: 8
- total_train_batch_size: 8
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.05
- num_epochs: 2.0

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 0.1052        | 0.2676 | 100  | 0.1041          |
| 0.0961        | 0.5351 | 200  | 0.0988          |
| 0.1051        | 0.8027 | 300  | 0.0960          |
| 0.1001        | 1.0696 | 400  | 0.0947          |
| 0.1001        | 1.3371 | 500  | 0.0935          |
| 0.0831        | 1.6047 | 600  | 0.0931          |
| 0.0652        | 1.8722 | 700  | 0.0928          |


### Framework versions

- PEFT 0.18.1
- Transformers 4.57.1
- Pytorch 2.6.0+cu124
- Datasets 4.0.0
- Tokenizers 0.22.2