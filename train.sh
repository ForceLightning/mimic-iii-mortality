#!/usr/bin/env bash
#
# Runs training for BioClinicalBERT on MIMIC-III mortality dataset.

function BERT_EHR() {
  echo "BERT + EHR seed $1"
  TOKENIZERS_PARALLELISM=true uv run --env-file .env -- python -m Main_modality_model \
    --model_type BERT_EHR \
    --data_type EHR_AND_REPORT \
    --data_dir ./data \
    --checkpoints_dir ./checkpoints/ \
    --batch_size 16 \
    --num_epoch 10 \
    --lr 1e-5 \
    --output_l1_dim 256 \
    --output_l2_dim 48 \
    --bert_model_str "emilyalsentzer/Bio_ClinicalBERT" \
    --seed_everything $1
}

function BERT_EHR_TCN() {
  echo "BERT + EHR + TCN seed $1"
  TOKENIZERS_PARALLELISM=true uv run --env-file .env -- python -m Main_modality_model \
    --model_type BERT_EHR \
    --data_type EHR_AND_REPORT \
    --data_dir ./data \
    --checkpoints_dir ./checkpoints/ \
    --batch_size 16 \
    --num_epoch 10 \
    --lr 1e-5 \
    --output_l1_dim 256 \
    --output_l2_dim 48 \
    --bert_use_temporal_conv \
    --seed_everything $1
}

function BERT_custom_model() {
  echo "BERT (no EHR) seed $1, $2 model"
  TOKENIZERS_PARALLELISM=true uv run --env-file .env -- python -m Main_modality_model \
    --model_type BERT \
    --data_type REPORT \
    --data_dir ./data \
    --checkpoints_dir ./checkpoints/ \
    --batch_size 16 \
    --num_epoch 10 \
    --lr 1e-5 \
    --output_l1_dim 256 \
    --output_l2_dim 48 \
    --bert_model_str $2 \
    --seed_everything $1
}

function BioclinicalBERT_Embeddings_TCN() {
  echo "Bioclinical_BERT embeddings w/ TCN + EHR seed $1"
  TOKENIZERS_PARALLELISM=true uv run --env-file .env -- python -m Main_modality_model \
    --model_type TCN \
    --data_type BCB_EMB_EHR \
    --data_dir ./data \
    --checkpoints_dir ./checkpoints/ \
    --batch_size 16 \
    --num_epoch 10 \
    --lr 3e-4 \
    --output_l1_dim 256 \
    --output_l2_dim 48 \
    --seed_everything $1
}

function phenotyping_BERT() {
  echo "BERT (phenotyping) seed $1"
  TOKENIZERS_PARALLELISM=true uv run --env-file .env -- python -m Main_modality_model \
    --model_type BERT \
    --data_type PHENOTYPE_REPORT \
    --data_dir ./data \
    --checkpoints_dir ./checkpoints/ \
    --batch_size 16 \
    --num_epoch 10 \
    --lr 3e-4 \
    --output_l1_dim 256 \
    --output_l2_dim 48 \
    --seed_everything $1
}

function phenotyping_BERT_EHR() {
  echo "BERT (phenotyping + EHR) seed $1"
  TOKENIZERS_PARALLELISM=true uv run --env-file .env -- python -m Main_modality_model \
    --model_type BERT \
    --data_type PHENOTYPE_EHR_AND_REPORT \
    --data_dir ./data \
    --checkpoints_dir ./checkpoints/ \
    --batch-size 16 \
    --num_epoch 10 \
    --lr 3e-4 \
    --output_l1_dim 256 \
    --output_l2_dim 48 \
    --seed_everything $1
}

function loop_3407_3409() {
  for seed in {3407..3409}; do
    $1 $seed
  done
}

function loop_3407_3409_params() {
  for seed in {3407..3409}; do
    $1 $seed $2
  done
}

# loop_3407_3409_params BERT_custom_model "google-bert/bert-base-uncased"
# loop_3407_3409_params BERT_custom_model "dmis-lab/biobert-v1.1"
# loop_3407_3409_params BERT_custom_model "emilyalsentzer/Bio_ClinicalBERT"
#
# loop_3407_3409 BERT_EHR

# BERT_EHR 3408
# loop_3407_3409 BioclinicalBERT_Embeddings_TCN

loop_3407_3409 phenotyping_BERT
loop_3407_3409 phenotyping_BERT_EHR
