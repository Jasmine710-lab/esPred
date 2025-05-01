# eSPred: Explainable scRNA-seq Prediction via Customized Foundation Models and Pathway-Aware Fine-tuning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

eSPred is a framework that enhances foundation models for single-cell RNA sequencing (scRNA-seq) analysis through cell-type-informed input strategies and biological pathway integration to achieve superior subject-level predictions and interpretability.

## Overview

eSPred integrates three key components to bridge cellular heterogeneity with clinical outcome prediction:

1. **Cell-type-aware Grouping**: A customized pre-training strategy that groups cells of the same type to generate more informative cell embeddings
2. **Pathway-guided Decoder**: A biologically-informed neural architecture that integrates Reactome pathway knowledge during fine-tuning
3. **Hierarchical Classification**: A two-step approach that aggregates cell-level predictions to subject-level outcomes

## Installation

### Dependencies

```bash
Option 1: Install via requirements.txt
pip install -r requirements.txt

Option 2: Follow scGPT installation instructions
See scGPT.README.md for detailed instructions
```

## Usage
### 1. Pre-training with Cell-type-aware Grouping
This step enhances the foundation model by incorporating cell-type information through strategic grouping.
python pretrain.py \
    --data-source $DATASET \
    --save-dir ./save/eval-$(date +%b%d-%H-%M-%Y) \
    --load-model $CHECKPOINT \
    --max-seq-len $MAX_LENGTH \
    --batch-size $per_proc_batch_size \
    --eval-batch-size $(($per_proc_batch_size * 2)) \
    --epochs 100 \
    --log-interval $LOG_INTERVAL \
    --trunc-by-sample \
    --no-cls \
    --no-cce \
    --fp16

### 2. Fine-tuning with Pathway Integration
This step integrates biological pathway knowledge into the model during fine-tuning.
python finetune.py \
  --dataset_name covid \
  --load_model /path/to/pretrained_model \
  --output_dir ./results/covid \
  --epochs 50 \
  --batch_size 32 \
  --lr 1e-4 \
  --layer_size 128 \
  --n_hidden 2 \
  --dropout 0.2 \
  --freeze
