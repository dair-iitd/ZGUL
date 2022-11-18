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
GPU=${1:-0}
MODEL=${2:-bert-base-multilingual-cased}
DATA_DIR=${3:-"$REPO/data/"}
OUT_DIR=${4:-"$REPO/outputs/"}

export CUDA_VISIBLE_DEVICES=$GPU

TASK='udpos'
LANGS="qpm,hsb,orv,cu"
#LANGS="cu"
TRAIN_LANGS="en,ru,cs"

NUM_EPOCHS=1
MAX_LENGTH=128
LR=1e-4
BPE_DROP=0
ADAPTER_LANG="en,ru,cs"
LANG_ADAPTER_NAME="en/wiki@ukp,ru/wiki@ukp,cs/wiki@ukp"
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
for SEED in 1;
do
OUTPUT_DIR="output/udpos/bert-base-multilingual-cased-LR1e-4-epoch5-MaxLen128-TrainLangen,ru,cs-Rf3_en,ru,cs_s1_ours_layerwise_dot/checkpoint-best-5/"

python third_party/run_test_pos.py \
  --predict_save_prefix "" \
  --per_gpu_eval_batch_size  8 \
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
  --seed 1 \
  --predict_langs $LANGS \
  --train_langs $TRAIN_LANGS \
  --log_file $OUTPUT_DIR/train.log \
  --eval_all_checkpoints \
  --eval_patience -1 \
  --overwrite_output_dir \
  --test_adapter \
  --adapter_config pfeiffer \
  --task_name "pos" \
  --lang_adapter_config pfeiffer \
  --language $ADAPTER_LANG \
  --l2v \
  --load_lang_adapter $LANG_ADAPTER_NAME \
  --rf 4 
done
