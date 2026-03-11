# WhiSSDapt

**WhiSSDapt** is a Whisper-based framework for **automatic sentence stress detection**.
The model adapts the pretrained **Whisper encoder–decoder architecture** by learning an adaptive fusion of internal layer representations to detect stressed words in spoken utterances.

Unlike prior approaches that rely on representations from a **single fixed layer**, WhiSSDapt learns **weighted combinations of encoder and decoder layers**, allowing the model to automatically select the most informative representations for stress prediction. This approach improves stress detection performance across both **naturally spoken and synthetic datasets**. 

The Whisper backbone remains **frozen**, and only a small set of task-specific parameters are trained.

---

# Repository Structure

```
WhiSSDapt/
│
├── model/
│   ├── model.py
│   └── model_layer_concat.py
│
├── training/
│   ├── train.py
│   ├── trainer.py
│   ├── data_loader.py
│   ├── data_collator.py
│   ├── processor.py
│   ├── metrics.py
│   └── layer_weights.py
│
├── run_training.sh
└── README.md
```

---

# Model Implementations

## `model/model.py`

Main implementation of **WhiSSDapt**.

Key characteristics:

* Frozen **Whisper-small.en** backbone
* Learnable **encoder layer weights**
* Learnable **decoder layer weights**
* Temperature-scaled softmax normalization
* Additional **decoder block**
* Two-layer feedforward **classification head**

The model learns an **adaptive weighted fusion of Whisper encoder and decoder layers** instead of relying on a fixed layer.

---

## `model/model_layer_concat.py`

Alternative model used for **ablation experiments**.

This variant performs **fixed layer fusion** by concatenating selected encoder layers and projecting them back to the Whisper hidden dimension before passing them to the decoder block. 

This configuration was used to evaluate fixed-layer combinations described in the paper.

---

# Configuration Notes
* **Weighted encoder + fixed decoder fusion** configuration is already included inside 'model.py' but commented our for convenience

This can be enabled directly in the model file for ablation experiments.

---

# Training

Training is launched using:

```
bash run_training.sh
```

The script internally runs:

```
torchrun -m whistress.training.train
```

This uses the HuggingFace training framework with a custom trainer that contains both token-level and word-level evaluation.

---

# Training Arguments

The training script accepts the following arguments.

| Argument                      | Description                                                                        |
| ----------------------------- | ---------------------------------------------------------------------------------- |
| `--model_path`                | Path to a trained model used for evaluation                                        |
| `--dataset_path`              | Directory used to store or load preprocessed datasets                              |
| `--output_path`               | Directory where training results are saved                                         |
| `--transcription_column_name` | Dataset transcription column (`transcription` or `aligned_whisper_transcriptions`) |
| `--dataset_train`             | Dataset used for training                                                          |
| `--dataset_eval`              | Dataset used for evaluation                                                        |
| `--is_train`                  | Whether to train (`true`) or run evaluation only (`false`)                         |

---

# Example Training Command

```
torchrun --nproc_per_node=1 -m whistress.training.train \
    --dataset_path TinyStress-15K-preprocessed \
    --transcription_column_name transcription \
    --dataset_train tinyStress-15K \
    --dataset_eval tinyStress-15K \
    --is_train true
```

The same configuration is provided in:

```
run_training.sh
```

---

# Output

After training, the following files are saved:

```
training_results/
│
├── classifier.pt
├── additional_decoder_block.pt
├── layer_weights.pt
├── metadata.json
└── training_args.json
```

These contain the trained task-specific parameters while keeping the Whisper backbone external.

---

# Layer Weight Analysis

The script

```
training/layer_weights.py
```

can be used to inspect the **learned layer weight distributions** after training.
This provides insight into which encoder and decoder layers contribute most to stress detection.
