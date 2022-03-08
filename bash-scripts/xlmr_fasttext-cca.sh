#!/bin/bash
TRAINENV="path/to/venv"

# for tracking with Weights & Biases
export WANDB_PROJECT="continued-pretraining"
export WANDB_MODE="offline"

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

source ${TRAINENV}/bin/activate

python3 ../cont-pretraining/run_intermediate_training.py \
  --model_name_or_path 'xlm-roberta-base' \
  --train_data_files '/path/to/train-data/wiki_txt_40/??.txt' \
  --eval_data_file '/path/to/train-data/validation.txt' \
  --output_dir '/path/to/checkpoints/xlmr-fasttext-cca' \
  --train_swe_files '/path/to/train-data/fasttext/??.vec' \
  --data_cache_dir '/path/to/train-data-cache/' \
  --overwrite_output_dir \
  --do_train \
  --do_eval \
  --max_steps 7500 \
  --per_device_train_batch_size 4 \
  --per_device_swe_batch_size 64 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 16 \
  --evaluation_strategy 'steps' \
  --eval_steps 500 \
  --save_strategy "steps" \
  --save_steps 500 \
  --save_total_limit 1 \
  --load_best_model_at_end \
  --metric_for_best_model "eval_loss" \
  --block_size 128 \
  --run_name "XLM-R + MLM + fasttext/CCA"  # for Weights & Biases
