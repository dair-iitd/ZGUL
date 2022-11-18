# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors,
# The HuggingFace Inc. team, and The XTREME Benchmark Authors.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""Fine-tuning models for NER and POS tagging."""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy
import torch
from transformers.adapters.composition import Fuse, Stack
from seqeval.metrics import precision_score, recall_score, f1_score
from tensorboardX import SummaryWriter
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from utils_tag import convert_examples_to_features
from utils_tag import get_labels
from utils_tag import read_examples_from_file
import pdb
import json

from transformers import (
  AdamW,
  get_linear_schedule_with_warmup,
  WEIGHTS_NAME,
  AutoConfig,
  AutoModelForTokenClassification,
  AutoTokenizer,
  HfArgumentParser,
  MultiLingAdapterArguments,
  AdapterConfig,
  AdapterType,
)
#from xlm import XLMForTokenClassification

#LANG2ID = {'en':0, 'hi':1, 'ar':2, 'mr':3, 'ta':4, 'bn':5, 'bho':6, 'amh':7, 'hau':8, 'ibo':9, 'kin':10, 'lug':11, 'luo':12, 'pcm':13, 'swa':14, 'wol':15, 'yor':16, 'bh':6, 'as':1, 'or':1, 'ur':21}
l2l_map={'en':'eng', 'is':'isl', 'de':'deu','fo':'fao', 'got':'got', 'gsw':'gsw', 'da':'dan', 'no':'nor', 'ru':'rus', 'cs':'ces', 'qpm':'bul', 'hsb':'hsb', 'orv':'chu', 'cu':'chu', 'bg':'bul', 'uk':'ukr', 'be':'bel','am':'amh','sw':'swa','wo':'wol'}

with open("lang2id.json", "r") as f:
  LANG2ID = json.load(f)

for k,v in l2l_map.items():
  LANG2ID[k] = LANG2ID[v]


logger = logging.getLogger(__name__)

def set_seed(args):
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  if args.n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)

def train(args, train_dataset, model, tokenizer, labels, pad_token_label_id, lang_adapter_names, task_name, adap_ids, lang2id=LANG2ID, wandb=None):
  """Train the model."""
  if args.local_rank in [-1, 0]:
    tb_writer = SummaryWriter()

  args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
  train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
  train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

  if args.max_steps > 0:
    t_total = args.max_steps
    args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
  else:
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

  # Prepare optimizer and schedule (linear warmup and decay)
  no_decay = ["bias", "LayerNorm.weight"]
  optimizer_grouped_parameters = [
    {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
     "weight_decay": args.weight_decay},
    {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
  ]
  #pdb.set_trace()
  optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
  logging.info([n for (n, p) in model.named_parameters() if p.requires_grad])
  scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1*t_total, num_training_steps=t_total)
  if args.fp16:
    try:
      from apex import amp
    except ImportError:
      raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

  # multi-gpu training (should be after apex fp16 initialization)
  if args.n_gpu > 1:
    model = torch.nn.DataParallel(model)

  # Distributed training (should be after apex fp16 initialization)
  if args.local_rank != -1:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                              output_device=args.local_rank,
                              find_unused_parameters=True)

  # Train!
  logger.info("***** Running training *****")
  logger.info("  Num examples = %d", len(train_dataset))
  logger.info("  Num Epochs = %d", args.num_train_epochs)
  logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
  logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size * args.gradient_accumulation_steps * (
          torch.distributed.get_world_size() if args.local_rank != -1 else 1))
  logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
  logger.info("  Total optimization steps = %d", t_total)

  best_score = 0.0
  best_checkpoint = None
  patience = 0
  global_step = 0
  tr_loss, logging_loss = 0.0, 0.0
  model.zero_grad()
  train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
  set_seed(args) # Add here for reproductibility (even between python 2 and 3)

  cur_epoch = 0
  for _ in train_iterator:
    epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
    cur_epoch += 1
    for step, batch in enumerate(epoch_iterator):
      batch = tuple(t.to(args.device) for t in batch if t is not None)
      #pdb.set_trace()
      #print(batch[-1])
      if args.l2v:
        batch_size_ = batch[0].shape[0]
        inputs = {"input_ids": torch.cat((batch[0],batch[-1], adap_ids.repeat(batch_size_,1).to('cuda')),1),
            "attention_mask": batch[1],            
            "labels": batch[3]}
        #pdb.set_trace()
      else:
        inputs = {"input_ids": batch[0],
              "attention_mask": batch[1],
              "labels": batch[3]}
      if args.model_type != "distilbert":
        # XLM and RoBERTa don"t use segment_ids
        inputs["token_type_ids"] = batch[2] if args.model_type in ["bert", "xlnet"] else None

      if args.model_type == "xlm":
        pdb.set_trace()
        inputs["langs"] = batch[4]
      outputs,_ = model(**inputs)
      #pdb.set_trace()
      loss = outputs[0]

      if args.n_gpu > 1:
        # mean() to average on multi-gpu parallel training
        loss = loss.mean()
      if args.gradient_accumulation_steps > 1:
        loss = loss / args.gradient_accumulation_steps

      if args.fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
          scaled_loss.backward()
      else:
        loss.backward()
      tr_loss += loss.item()
      if (step + 1) % args.gradient_accumulation_steps == 0:
        if args.fp16:
          torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
        else:
          torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        optimizer.step()  # Update learning rate schedule
        scheduler.step()
        model.zero_grad()
        global_step += 1

        if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
          # Log metrics
          if args.local_rank == -1 and args.evaluate_during_training:
            # Only evaluate on single GPU otherwise metrics may not average well
            results, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="dev", lang=args.train_langs, adap_ids=adap_ids, lang2id=lang2id, lang_adapter_names=lang_adapter_names, task_name=task_name)
            for key, value in results.items():
              tb_writer.add_scalar("eval_{}".format(key), value, global_step)
          tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
          tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
          logging_loss = tr_loss
        #pdb.set_trace()
        # if global_step == 1:
        #   output_dir = os.path.join(args.output_dir, "checkpoint-best-0")
        #   if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)
        #   model_to_save = model.module if hasattr(model, "module") else model
        #   model_to_save.save_all_adapters(output_dir)
        #   model_to_save.save_all_adapter_fusions(output_dir)
        #   model_to_save.save_pretrained(output_dir)
        #   tokenizer.save_pretrained(output_dir)

        if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
          if args.save_only_best_checkpoint:
            result, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="dev", prefix=global_step, lang=args.train_langs, adap_ids=adap_ids, lang2id=lang2id, lang_adapter_names=lang_adapter_names, task_name=task_name)
            if result["f1"] > best_score:
              logger.info("result['f1']={} > best_score={}".format(result["f1"], best_score))
              #pdb.set_trace()
              best_score = result["f1"]
              # Save the best model checkpoint
              # r1_mr, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="dev", prefix=global_step, lang='mr', adap_ids=adap_ids, lang2id=lang2id, lang_adapter_names=lang_adapter_names, task_name=task_name)
              # s1_mr = r1_mr["f1"]
              # print("Mr "+str(s1_mr))

              # r1_ta, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="dev", prefix=global_step, lang='ta', adap_ids=adap_ids, lang2id=lang2id, lang_adapter_names=lang_adapter_names, task_name=task_name)
              # s1_ta = r1_ta["f1"]
              # print("Ta "+str(s1_ta))

              # r1_bn, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="dev", prefix=global_step, lang='bn', adap_ids=adap_ids, lang2id=lang2id, lang_adapter_names=lang_adapter_names, task_name=task_name)
              # s1_bn = r1_bn["f1"]
              # print("Bn "+str(s1_bn))

              # avg_mr = (s1_mr + s1_ta + s1_bn)/3
              # print("Avg "+str(avg_mr))
              temp_ = global_step
              if cur_epoch <= 5:
                output_dir = os.path.join(args.output_dir, "checkpoint-best-5")
              elif cur_epoch <= 10:
                output_dir = os.path.join(args.output_dir, "checkpoint-best-10")
              else:
                output_dir = os.path.join(args.output_dir, "checkpoint-best-15")
              #best_checkpoint = output_dir
              if not os.path.exists(output_dir):
                os.makedirs(output_dir)
              # Take care of distributed/parallel training
              model_to_save = model.module if hasattr(model, "module") else model
              #pdb.set_trace()
              if args.do_save_adapters:
                #print("PASS1")
                model_to_save.save_all_adapters(output_dir)
              if args.do_save_adapter_fusions:
                #print("PASS2")
                model_to_save.save_all_adapter_fusions(output_dir)
              #if args.do_save_full_model:
                #print("PASS3")
                #model_to_save.save_pretrained(output_dir)
              model_to_save.save_pretrained(output_dir)
              tokenizer.save_pretrained(output_dir)
              torch.save(args, os.path.join(output_dir, "training_args.bin"))
              logger.info("Saving the best model checkpoint to %s", output_dir)
              logger.info("Reset patience to 0")
              patience = 0
            else:
              patience += 1
              logger.info("Hit patience={}".format(patience))
              if args.eval_patience > 0 and patience > args.eval_patience:
                logger.info("early stop! patience={}".format(patience))
                epoch_iterator.close()
                train_iterator.close()
                if args.local_rank in [-1, 0]:
                  tb_writer.close()
                return global_step, tr_loss / global_step

      if args.max_steps > 0 and global_step > args.max_steps:
        epoch_iterator.close()
        break
    if args.max_steps > 0 and global_step > args.max_steps:
      train_iterator.close()
      break

  if args.local_rank in [-1, 0]:
    tb_writer.close()

  return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, labels, pad_token_label_id, mode, prefix="", lang="en", adap_ids=None, lang2id=None, print_result=True, adapter_weight=None, lang_adapter_names=None, task_name=None, calc_weight_step=0):
  eval_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode=mode, lang=lang, lang2id=lang2id)

  args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
  # Note that DistributedSampler samples randomly
  eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
  eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

  # multi-gpu evaluate
  if args.n_gpu > 1:
    model = torch.nn.DataParallel(model)
  # Eval!
  logger.info("***** Running evaluation %s in %s *****" % (prefix, lang))
  logger.info("  Num examples = %d", len(eval_dataset))
  logger.info("  Batch size = %d", args.eval_batch_size)
  eval_loss = 0.0
  nb_eval_steps = 0
  preds = None
  out_label_ids = None
  model.eval()
  for batch in tqdm(eval_dataloader, desc="Evaluating"):
    batch = tuple(t.to(args.device) for t in batch)

    if calc_weight_step > 0:
      pdb.set_trace()
      adapter_weight = calc_weight_multi(args, model, batch, lang_adapter_names, task_name, adapter_weight, 0)
    with torch.no_grad():
      if args.l2v:
        batch_size_ = batch[0].shape[0]
        inputs = {"input_ids": torch.cat((batch[0],batch[-1], adap_ids.repeat(batch_size_,1).to('cuda')),1),
            "attention_mask": batch[1],            
            "labels": batch[3]}
        
      else:
        inputs = {"input_ids": batch[0],
              "attention_mask": batch[1],
              "labels": batch[3]}

      if args.model_type != "distilbert":
        # XLM and RoBERTa don"t use segment_ids
        inputs["token_type_ids"] = batch[2] if args.model_type in ["bert", "xlnet"] else None
      if args.model_type == 'xlm':
        inputs["langs"] = batch[4]
      outputs,_ = model(**inputs)
      tmp_eval_loss, logits = outputs[:2]

      if args.n_gpu > 1:
        # mean() to average on multi-gpu parallel evaluating
        tmp_eval_loss = tmp_eval_loss.mean()

      eval_loss += tmp_eval_loss.item()
    nb_eval_steps += 1
    if preds is None:
      preds = logits.detach().cpu().numpy()
      out_label_ids = inputs["labels"].detach().cpu().numpy()
    else:
      preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
      out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

  if nb_eval_steps == 0:
    results = {k: 0 for k in ["loss", "precision", "recall", "f1"]}
  else:
    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=2)

    label_map = {i: label for i, label in enumerate(labels)}

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
      for j in range(out_label_ids.shape[1]):
        if out_label_ids[i, j] != pad_token_label_id:
          out_label_list[i].append(label_map[out_label_ids[i][j]])
          preds_list[i].append(label_map[preds[i][j]])

    results = {
      "loss": eval_loss,
      "precision": precision_score(out_label_list, preds_list),
      "recall": recall_score(out_label_list, preds_list),
      "f1": f1_score(out_label_list, preds_list)
    }

  if print_result:
    logger.info("***** Evaluation result %s in %s *****" % (prefix, lang))
    for key in sorted(results.keys()):
      logger.info("  %s = %s", key, str(results[key]))
  return results, preds_list


def load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode, lang, lang2id=LANG2ID, few_shot=-1):
  # Make sure only the first process in distributed training process
  # the dataset, and the others will use the cache
  if args.local_rank not in [-1, 0] and not evaluate:
    torch.distributed.barrier()

  # Load data features from cache or dataset file
  bpe_dropout = args.bpe_dropout
  if mode != 'train': bpe_dropout = 0
  if bpe_dropout > 0:
    cached_features_file = os.path.join(args.data_dir, "cached_{}_{}_{}_{}_drop{}".format(mode, lang,
      list(filter(None, args.model_name_or_path.split("/"))).pop(),
      str(args.max_seq_length), bpe_dropout))
  else:
    cached_features_file = os.path.join(args.data_dir, "cached_{}_{}_{}_{}".format(mode, lang,
      list(filter(None, args.model_name_or_path.split("/"))).pop(),
      str(args.max_seq_length)))
  if os.path.exists(cached_features_file) and not args.overwrite_cache:
    logger.info("Loading features from cached file %s", cached_features_file)
    features = torch.load(cached_features_file)
  else:
    langs = lang.split(',')
    logger.info("all languages = {}".format(lang))
    features = []
    for lg in langs:
      data_file = os.path.join(args.data_dir, lg, "{}.{}".format(mode, args.model_name_or_path))
      logger.info("Creating features from dataset file at {} in language {}".format(data_file, lg))
      examples = read_examples_from_file(data_file, lg, LANG2ID)
      features_lg = convert_examples_to_features(examples, labels, args.max_seq_length, tokenizer,
                          cls_token_at_end=bool(args.model_type in ["xlnet"]),
                          cls_token=tokenizer.cls_token,
                          cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                          sep_token=tokenizer.sep_token,
                          sep_token_extra=bool(args.model_type in ["roberta", "xlmr"]),
                          pad_on_left=bool(args.model_type in ["xlnet"]),
                          pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                          pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
                          pad_token_label_id=pad_token_label_id,
                          lang=lg,
                          bpe_dropout=bpe_dropout,
                          )
      features.extend(features_lg)
    if args.local_rank in [-1, 0]:
      logger.info("Saving features into cached file {}, len(features)={}".format(cached_features_file, len(features)))
      torch.save(features, cached_features_file)

  # Make sure only the first process in distributed training process
  # the dataset, and the others will use the cache
  if args.local_rank == 0 and not evaluate:
    torch.distributed.barrier()

  if few_shot > 0 and mode == 'train':
    logger.info("Original no. of examples = {}".format(len(features)))
    features = features[: few_shot]
    logger.info('Using few-shot learning on {} examples'.format(len(features)))

  # Convert to Tensors and build dataset
  all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
  all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
  all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
  all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
  if args.l2v and features[0].langs is not None:
    all_langs = torch.tensor([f.langs for f in features], dtype=torch.long)
    logger.info('all_langs[0] = {}'.format(all_langs[0]))
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_langs)
    #print(all_langs)
  else:
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
  return dataset


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_type: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    labels: str = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    data_dir: str = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    output_dir: str = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    max_seq_length: Optional[int] = field(
        default=128, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    do_train: Optional[bool] = field(default=False )
    do_eval: Optional[bool] = field(default=False )
    do_predict: Optional[bool] = field(default=False )
    do_adapter_predict: Optional[bool] = field(default=False )
    do_predict_dev: Optional[bool] = field(default=False )
    do_predict_train: Optional[bool] = field(default=False )
    init_checkpoint: Optional[str] = field(default=None )
    evaluate_during_training: Optional[bool] = field(default=False )
    do_lower_case: Optional[bool] = field(default=False )
    few_shot: Optional[int] = field(default=-1 )
    per_gpu_train_batch_size: Optional[int] = field(default=8)
    per_gpu_eval_batch_size: Optional[int] = field(default=8)
    gradient_accumulation_steps: Optional[int] = field(default=1)
    learning_rate: Optional[float] = field(default=5e-5)
    weight_decay: Optional[float] = field(default=0.0)
    adam_epsilon: Optional[float] = field(default=1e-8)
    max_grad_norm: Optional[float] = field(default=1.0)
    num_train_epochs: Optional[float] = field(default=3.0)
    max_steps: Optional[int] = field(default=-1)
    save_steps: Optional[int] = field(default=-1)
    warmup_steps: Optional[int] = field(default=0)
    logging_steps: Optional[int] = field(default=50)
    save_only_best_checkpoint: Optional[bool] = field(default=False)
    eval_all_checkpoints: Optional[bool] = field(default=False)
    no_cuda: Optional[bool] = field(default=False)
    overwrite_output_dir: Optional[bool] = field(default=False)
    overwrite_cache: Optional[bool] = field(default=False)
    seed: Optional[int] = field(default=42)
    fp16: Optional[bool] = field(default=False)
    fp16_opt_level: Optional[str] = field(default="O1")
    local_rank: Optional[int] = field(default=-1)
    server_ip: Optional[str] = field(default="")
    server_port: Optional[str] = field(default="")
    predict_langs: Optional[str] = field(default="en")
    train_langs: Optional[str] = field(default="en")
    log_file: Optional[str] = field(default=None)
    eval_patience: Optional[int] = field(default=-1)
    bpe_dropout: Optional[float] = field(default=0)
    do_save_adapter_fusions: Optional[bool] = field(default=False)
    do_save_full_model: Optional[bool] = field(default=False)
    do_save_adapters: Optional[bool] = field(default=False)
    task_name: Optional[str] = field(default="ner")
    l2v: Optional[bool] = field(default=False)
    madx2: Optional[bool] = field(default=False)
    predict_task_adapter: Optional[str] = field(default=None)
    predict_lang_adapter: Optional[str] = field(default=None)
    test_adapter: Optional[bool] = field(default=False)
    rf: Optional[int] = field(default=4)
    adapter_weight: Optional[str] = field(default=None)

    calc_weight_step: Optional[int] = field(default=0)
    predict_save_prefix: Optional[str] = field(default=None)

def main():
  parser = argparse.ArgumentParser()

  parser = HfArgumentParser((ModelArguments, MultiLingAdapterArguments))
  args, adapter_args = parser.parse_args_into_dataclasses()

  if os.path.exists(args.output_dir) and os.listdir(
      args.output_dir) and args.do_train and not args.overwrite_output_dir:
    raise ValueError(
      "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
        args.output_dir))

  # Setup distant debugging if needed
  if args.server_ip and args.server_port:
    import ptvsd
    print("Waiting for debugger attach")
    ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
    ptvsd.wait_for_attach()

  # Setup CUDA, GPU & distributed training
  if args.local_rank == -1 or args.no_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
  else:
  # Initializes the distributed backend which sychronizes nodes/GPUs
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="nccl")
    args.n_gpu = 1
  args.device = device

  # Setup logging
  logging.basicConfig(handlers = [logging.FileHandler(args.log_file), logging.StreamHandler()],
                      format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                      datefmt = '%m/%d/%Y %H:%M:%S',
                      level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
  logging.info("Input args: %r" % args)
  logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
           args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

  # Set seed
  set_seed(args)

  # Prepare NER/POS task
  labels = get_labels(args.labels)
  num_labels = len(labels)
  # Use cross entropy ignore index as padding label id
  # so that only real label ids contribute to the loss later
  pad_token_label_id = CrossEntropyLoss().ignore_index

  # Load pretrained model and tokenizer
  # Make sure only the first process in distributed training loads model/vocab
  if args.local_rank not in [-1, 0]:
    torch.distributed.barrier()

  config = AutoConfig.from_pretrained(
      args.config_name if args.config_name else args.model_name_or_path,
      num_labels=num_labels,
      cache_dir=args.cache_dir,
  )
  args.model_type = config.model_type
  tokenizer = AutoTokenizer.from_pretrained(
      args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
      do_lower_case=args.do_lower_case,
      cache_dir=args.cache_dir,
      use_fast=False,
  )

  
  if args.init_checkpoint:
    logger.info("loading from init_checkpoint={}".format(args.init_checkpoint))
    model = AutoModelForTokenClassification.from_pretrained(
        args.init_checkpoint,
        config=config,
        cache_dir=args.cache_dir,
    )
  else:
    logger.info("loading from existing model {}".format(args.model_name_or_path))
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir,
    )
  args.do_save_full_model= (not adapter_args.train_adapter)
  args.do_save_adapters=adapter_args.train_adapter
  if args.do_save_adapters:
      logging.info('save adapters')
      logging.info(adapter_args.train_adapter)
  if args.do_save_full_model:
      logging.info('save model')
  # Setup adapters

  cpg_name = "cpg"
  task_name="ner"
  if args.madx2:
    #pdb.set_trace()
    leave_out = [len(model.roberta.encoder.layer)-1]
    task_adapter_config = AdapterConfig.load(
             adapter_args.adapter_config,
            non_linearity=adapter_args.adapter_non_linearity,
            reduction_factor=args.rf,
            leave_out = leave_out,
    )
  else:
    task_adapter_config = AdapterConfig.load(
             adapter_args.adapter_config,
            non_linearity=adapter_args.adapter_non_linearity,
            reduction_factor=args.rf,
    )
  # load a set of language adapters
  #logging.info("loading lang adpater {}".format(adapter_args.load_lang_adapter))
  # resolve the language adapter config
  if args.madx2:
    #pdb.set_trace()
    lang_adapter_config = AdapterConfig.load(
        adapter_args.lang_adapter_config,
        non_linearity=adapter_args.lang_adapter_non_linearity,
        reduction_factor=adapter_args.lang_adapter_reduction_factor,
        leave_out = leave_out,
    )
  else:
    lang_adapter_config = AdapterConfig.load(
        adapter_args.lang_adapter_config,
        non_linearity=adapter_args.lang_adapter_non_linearity,
        reduction_factor=adapter_args.lang_adapter_reduction_factor,
    )
  # load the language adapter from Hub
  languages = adapter_args.language.split(",")
  #adapter_names = adapter_args.load_lang_adapter.split(",")
  #assert len(languages) == len(adapter_names)
  lang_adapter_names = []
  #pdb.set_trace()
  for language in languages:
    print(language)
    #pdb.set_trace()
    lang_adapter_name = model.load_adapter(
        language + "/wiki@ukp",
        # AdapterType.text_lang,
        # config=lang_adapter_config,
        load_as=language,
    )
    lang_adapter_names.append(lang_adapter_name)

  #pdb.set_trace()
  adapter_setup_ = Fuse('en','am','sw','wo')
  #adapter_setup_ = Fuse(lang_adapter_names)
  model.add_adapter_fusion(adapter_setup_)
  #model.add_cpg_adapter(cpg_name, AdapterType.text_task, config=task_adapter_config)
  model.add_adapter(task_name, config=task_adapter_config)
  #model.train_adapter_final([cpg_name, task_name], lang_adapter_names)
  model.train_adapter_fusion_TA([task_name], adapter_setup_)
  #pdb.set_trace()
  model.set_active_adapters([lang_adapter_names, task_name])

  if args.l2v:
    adap_ids = torch.tensor([LANG2ID[it] for it in lang_adapter_names])
  else:
    adap_ids = None
  # model, lang_adapter_names, task_name, adap_ids = setup_adapter(args, adapter_args, model, num_labels)
  logger.info("lang adapter names: {}".format(" ".join(lang_adapter_names)))

  lang2id = LANG2ID if args.l2v else None
  logger.info("Using lang2id = {}".format(lang2id))

  # Make sure only the first process in distributed training loads model/vocab
  if args.local_rank == 0:
    torch.distributed.barrier()
  model.to(args.device)
  #pdb.set_trace()
  logger.info("Training/evaluation parameters %s", args)

  # Training
  train_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode="train", lang=args.train_langs, lang2id=lang2id, few_shot=args.few_shot)
  global_step, tr_loss = train(args, train_dataset, model, tokenizer, labels, pad_token_label_id, lang_adapter_names, task_name, adap_ids, lang2id)
  logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

  # Saving best-practices: if you use default names for the model,
  # you can reload it using from_pretrained()
  if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    # Create output directory if needed
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
      os.makedirs(args.output_dir)

    # Save model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    # Take care of distributed/parallel training
    logger.info("Saving model checkpoint to %s", args.output_dir)
    model_to_save = model.module if hasattr(model, "module") else model
    if args.do_save_adapters:
      logging.info("Save adapter")
      model_to_save.save_all_adapters(args.output_dir)
    if args.do_save_adapter_fusions:
      logging.info("Save adapter fusion")
      model_to_save.save_all_adapter_fusions(args.output_dir)
    if args.do_save_full_model:
      logging.info("Save full model")
      model_to_save.save_pretrained(args.output_dir)

    tokenizer.save_pretrained(args.output_dir)

    # Good practice: save your training arguments together with the model
    torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

if __name__ == "__main__":
  main()
