#!/bin/bash
mkdir -p logs

# ======================================================
# TinyStress-15K (COMMENTED OUT)
# ======================================================
torchrun --nproc_per_node=1 -m whistress.training.train \
    --dataset_path TinyStress-15K-preprocessed \
    --transcription_column_name transcription \
    --dataset_train tinyStress-15K \
    --dataset_eval tinyStress-15K \
    --is_train true \
    2>&1 | tee logs/training_output_tinystress.txt


# ======================================================
# GER-stress
# ======================================================
# torchrun --nproc_per_node=1 -m whistress.training.train \
#     --dataset_path GER-stress-preprocessed-GT \
#     --transcription_column_name transcription \
#     --dataset_train GER-stress \
#     --dataset_eval GER-stress \
#     --is_train true \
#     2>&1 | tee logs/training_output_ger_stress_GT.txt


# ======================================================
# ITA-stress 
# ======================================================
# torchrun --nproc_per_node=1 -m whistress.training.train \
#     --dataset_path ITA-stress-preprocessed-GT \
#     --transcription_column_name transcription \
#     --dataset_train ITA-stress \
#     --dataset_eval ITA-stress \
#     --is_train true \
#     2>&1 | tee logs/training_output_ita_stress_GT.txt
