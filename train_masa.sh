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
OUT_DIR=${6:-"$REPO/output/"}

export CUDA_VISIBLE_DEVICES=$GPU
TASK='masa'
LANGS="hau,ibo,kin,lug,luo,pcm"
TRAIN_LANGS=$1

NUM_EPOCHS=15
MAX_LENGTH=128
LR=1e-4
BPE_DROP=0

ADAPTER_LANG=$2

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
  BATCH_SIZE=32
  GRAD_ACC=4
fi


DATA_DIR="$DATA_DIR/${TASK}/${TASK}_processed_maxlen${MAX_LENGTH}/"
#rm -r $CACHE_DATA

for SEED in 42;
do
    for RF in 4 3;
    do
    OUTPUT_DIR="$OUT_DIR/$TASK/${MODEL}-LR${LR}-epoch${NUM_EPOCHS}-MaxLen${MAX_LENGTH}-TrainLang${TRAIN_LANGS}-Rf${RF}_${ADAPTER_LANG}_s${SEED}_cosine_onehot"
    mkdir -p $OUTPUT_DIR
    python third_party/run_train_ner.py \
    --do_train \
    --do_eval \
    --data_dir $DATA_DIR \
    --model_type $MODEL_TYPE \
    --labels $DATA_DIR/labels.txt \
    --model_name_or_path $MODEL \
    --output_dir $OUTPUT_DIR \
    --max_seq_length  $MAX_LENGTH \
    --num_train_epochs $NUM_EPOCHS \
    --gradient_accumulation_steps $GRAD_ACC \
    --per_gpu_train_batch_size $BATCH_SIZE \
    --per_gpu_eval_batch_size 32 \
    --save_steps 150 \
    --seed $SEED \
    --learning_rate $LR \
    --do_predict \
    --predict_langs $LANGS \
    --train_langs $TRAIN_LANGS \
    --log_file $OUTPUT_DIR/train.log \
    --eval_all_checkpoints \
    --eval_patience 40 \
    --overwrite_output_dir \
    --train_adapter \
    --adapter_config pfeiffer \
    --task_name "pos" \
    --language $ADAPTER_LANG \
    --lang_adapter_config pfeiffer \
    --save_only_best_checkpoint $LC \
    --do_save_adapters \
    --do_save_adapter_fusions \
    --l2v \
    --rf $RF 
    done
done

