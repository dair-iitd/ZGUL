#!/bin/bash
# Copyright 2020 Google and DeepMind.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

REPO=$PWD
GPU=${3:-0}
MODEL=${4:-bert-base-multilingual-cased}
DATA_DIR=${5:-"$REPO/data/"}
OUT_DIR=${6:-"$REPO/outputs/"}

export CUDA_VISIBLE_DEVICES=$GPU

TASK='panx'
LANGS="as,bh,ta,mr"
#LANGS="no,da"
#LANGS="qpm,hsb,orv,cu"
TRAIN_LANGS="en,hi,bn,ur"

NUM_EPOCHS=1
MAX_LENGTH=128
LR=1e-4
BPE_DROP=0
ADAPTER_LANG="en,hi,bn,ar"
LANG_ADAPTER_NAME="en/wiki@ukp,hi/wiki@ukp,bn/wiki@ukp,ar/wiki@ukp"
LC=""
if [ $MODEL == "bert-base-multilingual-cased" ]; then
  MODEL_TYPE="bert"
elif [ $MODEL == "xlm-mlm-100-1280" ] || [ $MODEL == "xlm-mlm-tlm-xnli15-1024" ]; then
  MODEL_TYPE="xlm"
  LC=" --do_lower_case"
elif [ $MODEL == "xlm-roberta-large" ] || [ $MODEL == "xlm-roberta-base" ]; then
  MODEL_TYPE="xlmr"
fi

if [ $MODEL == "xlm-mlm-100-1280" ] || [ $MODEL == "xlm-roberta-large" ]; then
  BATCH_SIZE=2
  GRAD_ACC=16
else
  BATCH_SIZE=2
  GRAD_ACC=4
fi

DATA_DIR="$DATA_DIR/${TASK}/${TASK}_processed_maxlen${MAX_LENGTH}/ours"
RF=4
for SEED in 1 2 3;
do
OUTPUT_DIR="output/panx/bert-base-multilingual-cased-LR1e-4-epoch15-MaxLen128-TrainLangen,hi,bn,ur-Rf3_en,hi,bn,ar_s1_ours/checkpoint-best-10/"
OUTFILE="$1_$2_indic_mbert_1.txt"
python analysis/entropy/run_test_pos.py \
  --predict_save_prefix "" \
  --per_gpu_eval_batch_size  1 \
  --data_dir $DATA_DIR \
  --model_type $MODEL_TYPE \
  --labels $DATA_DIR/labels.txt \
  --model_name_or_path $MODEL \
  --output_dir $OUTPUT_DIR \
  --max_seq_length  $MAX_LENGTH \
  --num_train_epochs $NUM_EPOCHS \
  --gradient_accumulation_steps $GRAD_ACC \
  --per_gpu_train_batch_size $BATCH_SIZE \
  --save_steps 1000 \
  --seed $SEED \
  --predict_langs $LANGS \
  --train_langs $TRAIN_LANGS \
  --log_file $OUTPUT_DIR/train.log \
  --eval_all_checkpoints \
  --eval_patience -1 \
  --overwrite_output_dir \
  --test_adapter \
  --adapter_config pfeiffer \
  --task_name "ner" \
  --lang_adapter_config pfeiffer \
  --language $ADAPTER_LANG \
  --l2v \
  --outfile $OUTFILE \
  --calc_step $1 \
  --emea_lr $2 \
  --load_lang_adapter $LANG_ADAPTER_NAME \
  --rf $RF
done
