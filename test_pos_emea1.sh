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
GPU=${7:-0}
MODEL=${8:-bert-base-multilingual-cased}
DATA_DIR=${5:-"$REPO/data/"}
OUT_DIR=${6:-"$REPO/outputs/"}

export CUDA_VISIBLE_DEVICES=$GPU

TASK='udpos'
#LANGS="gsw"
#LANGS="be,uk,bg"
LANGS=$1
TRAIN_LANGS="en,is,de"

NUM_EPOCHS=1
MAX_LENGTH=128
LR=1e-4
BPE_DROP=0
ADAPTER_LANG="en,is,de"
#ADAPTER_LANG="en,ru,cs"
LANG_ADAPTER_NAME="en/wiki@ukp,am/wiki@ukp,sw/wiki@ukp"
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
SEED=1

RF=$4
OUTPUT_DIR="ckpt/germanic/bert-base-multilingual-cased-LR5e-5-epoch10-MaxLen128-BS32-TrainLangen,is,de-Rf3_en,is,de_s1_ours_1.0/checkpoint-best-5/"
#OUTPUT_DIR="submission_latest/slavic/bert-base-multilingual-cased-LR1e-4-epoch5-MaxLen128-TrainLangen,ru,cs-Rf3_en,ru,cs_s1_ours_layerwise_dot/checkpoint-best-5/"
#OUTPUT_DIR="output/udpos/bert-base-multilingual-cased-LR1e-4-epoch5-MaxLen128-TrainLangen,is,de-Rf3_en,is,de_s42_final/checkpoint-best-5/"
#OUTPUT_DIR="output/udpos/bert-base-multilingual-cased-LR1e-4-epoch5-MaxLen128-TrainLangen,is,de-Rf3_en,is,de_s42_slavic_latest/checkpoint-best-5/"
#OUTPUT_DIR="output/udpos/bert-base-multilingual-cased-LR1e-4-epoch5-MaxLen128-TrainLangen,is,de-Rf3_en,is,de_s42_maxpool/checkpoint-best-5/"
#OUTPUT_DIR="output/mlm/shiva/bert-base-multilingual-cased-LR1e-4-epoch10-MaxLen128-TrainLangen,amh,swa,wol-Rf${RF}_en_conll,am,sw,wo_s42_zgul_load_${LANGS}/checkpoint-best-10/"
#OUTPUT_DIR="output/mlm/masa/bert-base-multilingual-cased-LR1e-4-epoch10-MaxLen128-TrainLangen,amh,swa,wol-Rf4_en_conll,am,sw,wo_s42_lanvec_plus_load_${LANGS}_mlm/checkpoint-best-10/"
OUTFILE="$2_$3_masa_tied.txt"
python run_test_pos_em.py \
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
  --task_name "pos" \
  --lang_adapter_config pfeiffer \
  --language $ADAPTER_LANG \
  --l2v \
  --outfile $OUTFILE \
  --calc_step $2 \
  --emea_lr $3 \
  --load_lang_adapter $LANG_ADAPTER_NAME \
  --rf $RF
