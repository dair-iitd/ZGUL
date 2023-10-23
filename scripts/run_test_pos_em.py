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
import pycm
import numpy as np
import scipy
import torch
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
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
import pandas as pd

from transformers import (
  AdamW,
  get_linear_schedule_with_warmup,
  WEIGHTS_NAME,
  AutoConfig,
  AutoModelForTokenClassification,
  AutoTokenizer,
  BertTokenizer,
  HfArgumentParser,
  MultiLingAdapterArguments,
  AdapterConfig,
  AdapterType,
)
#from xlm import XLMForTokenClassification

#l2l_map={'en':'eng', 'is':'isl', 'de':'deu','fo':'fao', 'got':'got', 'gsw':'gsw', 'da':'dan', 'no':'nor', 'ru':'rus', 'cs':'ces', 'qpm':'bul', 'hsb':'hsb', 'orv':'chu', 'cu':'chu', 'bg':'bul', 'uk':'ukr', 'be':'bel'}
l2l_map={'en':'eng', 'is':'isl', 'de':'deu','fo':'fao', 'got':'got', 'gsw':'gsw', 'da':'dan', 'no':'nor', 'ru':'rus', 'cs':'ces', 'qpm':'bul', 'hsb':'hsb', 'orv':'chu', 'cu':'chu', 'bg':'bul', 'uk':'ukr', 'be':'bel','am':'amh','sw':'swa','wo':'wol'}

map_lid = {9:4, 7:0, 14:3, 15:0, 0:0, 8:4, 10:4, 11:4, 13:4, 19:0, 6:0, 31:0, 32:0, 33:0, 27:0, 28:0, 7:3, 0:0, 1:1, 5:2}

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

def isNormal(tensor_list):
  FLAG = True
  for t in tensor_list:
    try:
      x_, y_, z_ = t.shape[0], t.shape[1], t.shape[2]
      assert round(t.reshape(-1,z_).sum().item(), 2) == x_*y_
    except:
      FLAG = False
  return FLAG

def get_entropy(outputs, mode='sum'):
  final_kept_logits = outputs[-1]
  final_entropy = torch.nn.functional.softmax(final_kept_logits, dim=-1)*torch.nn.functional.log_softmax(final_kept_logits, dim=-1)
  if mode == 'sum':
    final_entropy = -final_entropy.sum() / final_kept_logits.size(0)
  else:
    final_entropy = -final_entropy.sum() / final_kept_logits.size(1)
  return round(final_entropy.item(),4)

def test_emea(args, inputs, model, batch, lang_adapter_names, task_name, adapter_weights):
  #pdb.set_trace()
  calc_step_ = args.calc_step
  emea_lr_ = args.emea_lr
  #pdb.set_trace()
  batch_, max_len_ = adapter_weights[0].shape[0]//2, adapter_weights[0].shape[1]
  for step_ in range(calc_step_):
    for w in adapter_weights: 
      try:
        assert w.requires_grad == True
      except:  
        w.requires_grad = True
    if step_ > 0:
      w1_, w2_ = [],[]
      for it in adapter_weights:
        z1,z2=torch.split(it,batch_,0)
        w1_.append(z1)
        w2_.append(z2)
      normed_adapter_weights1 = [torch.nn.functional.softmax(w[0], dim=-1) for w in w1_]
      normed_adapter_weights2 = [torch.nn.functional.softmax(w[0], dim=-1) for w in w2_]
      normed_adapter_weights = [torch.cat((z1.unsqueeze(0),z2.unsqueeze(0)),0) for z1,z2 in zip(normed_adapter_weights1, normed_adapter_weights2)]
      try:
        assert not isNormal(adapter_weights) and isNormal(normed_adapter_weights)
      except:
        pass
      inputs["adapter_weights"] = normed_adapter_weights
    else:
      assert isNormal(adapter_weights)
      inputs["adapter_weights"] = adapter_weights
      
    outputs, _ = model(**inputs)
    kept_logits = outputs[-1]
    #pdb.set_trace()
    entropy = torch.nn.functional.softmax(kept_logits, dim=1)*torch.nn.functional.log_softmax(kept_logits, dim=1)
    entropy = -entropy.sum() / kept_logits.size(0)
    #print(entropy)
    try:
      grads = torch.autograd.grad(entropy, adapter_weights)
    except:
      pdb.set_trace()
    for i in range(12):
      adapter_weights[i] = adapter_weights[i].data - emea_lr_*grads[i].data
  if calc_step_ > 0:
    normed_adapter_weights = [torch.nn.functional.softmax(w, dim=-1) for w in adapter_weights]
    inputs["adapter_weights"] = normed_adapter_weights
  else:
    inputs["adapter_weights"] = adapter_weights

  outputs, _  = model(**inputs)
  #ret_entropy = get_entropy(outputs, 'avg')
  #print(ret_entropy)
  return outputs


def classification_report_csv(args, report, lang):
    report_data = []
    lines = report.split('\n')
    #pdb.set_trace()
    for line in lines[2:-2]:
        if line.split() == []:
          continue
        row = {}
        row_data = line.split()
        if len(row_data) == 6:
          row_data.remove("avg")
        row['class'] = row_data[0]
        
        try:
          row['precision'] = float(row_data[1])
        except:
          pdb.set_trace()
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv(lang+"_"+str(args.emea_lr)+"_"+str(args.calc_step)+'_report.csv', index = False)


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
  weights1 = []
  masks = []
  weights_dict_fus = {}
  weights_dict_l2v = {}
  entr_list = []
  model.eval()
  for batch in tqdm(eval_dataloader, desc="Evaluating"):
    batch = tuple(t.to(args.device) for t in batch)

    if calc_weight_step > 0:
      pdb.set_trace()
      adapter_weight = calc_weight_multi(args, model, batch, lang_adapter_names, task_name, adapter_weight, 0)
    #with torch.no_grad():
    if args.l2v:
      batch_size_ = batch[0].shape[0]
      inputs = {"input_ids": torch.cat((batch[0],batch[-1], adap_ids.repeat(batch_size_,1).to('cuda:0')),1),
          "attention_mask": batch[1],            
          "labels": batch[3]}
      masks += [it for it in batch[1]]
      
    else:
      inputs = {"input_ids": batch[0],
            "attention_mask": batch[1],
            "labels": batch[3]}

    if args.model_type != "distilbert":
      # XLM and RoBERTa don"t use segment_ids
      inputs["token_type_ids"] = batch[2] if args.model_type in ["bert", "xlnet"] else None
    if args.model_type == 'xlm':
      inputs["langs"] = batch[4]

    outputs, adapter_weights = model(**inputs)

    #curr_entr = get_entropy(outputs)
    assert len(adapter_weights)==12 and adapter_weights[0] is not None
    #pdb.set_trace()
    outputs = test_emea(args, inputs, model, batch, lang_adapter_names, task_name, adapter_weights)
    entr_list.append(curr_entr)
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
    fout_list = []
    fpred_list = []
    
    for i in range(out_label_ids.shape[0]):
      for j in range(out_label_ids.shape[1]):
        if out_label_ids[i, j] != pad_token_label_id:
          out_label_list[i].append(label_map[out_label_ids[i][j]])
          preds_list[i].append(label_map[preds[i][j]])
          fout_list.append(label_map[out_label_ids[i][j]])
          fpred_list.append(label_map[preds[i][j]])

    results = {
      "loss": eval_loss,
      "precision": precision_score(out_label_list, preds_list),
      "recall": recall_score(out_label_list, preds_list),
      "f1": f1_score(out_label_list, preds_list, average="micro"),
      "macro": f1_score(out_label_list, preds_list, average="macro")
    }

  report = classification_report(out_label_list, preds_list)
  #pdb.set_trace()
  with open(lang+"_zgul_pred.txt", "w") as f_w:
    cnt = 0
    for sent in preds_list:
      cnt += 1
      write_str = "\n".join(["\t".join(("dummy", lab_)) for lab_ in sent])
      f_w.write(write_str+"\n\n")
      #if cnt == 100:
      #  break

  with open(lang+"_zgul_conf.json", "w") as f_w:
    json.dump(entr_list, f_w)
  #classification_report_csv(args, report, lang)

  if print_result:
    logger.info("***** Evaluation result %s in %s *****" % (prefix, lang))
    for key in sorted(results.keys()):
      logger.info("  %s = %s", key, str(results[key]))
    #cm = pycm.ConfusionMatrix(actual_vector=fout_list, predict_vector=fpred_list)
    #print(cm)
  with open(str(args.calc_step)+"_"+str(args.emea_lr)+"_"+lang+"_zgul.txt","a") as f:
    write_st = args.output_dir+" "+lang+" "+str(results["f1"])+"\n"
    f.write(write_st)
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
      #pdb.set_trace()
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
    outfile: Optional[str] = field(default=None)
    zgul_plus: Optional[bool] = field(default=False)
    calc_weight_step: Optional[int] = field(default=0)
    calc_step: Optional[int] = field(default=1)
    emea_lr: Optional[float] = field(default=1.0)
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
  #config = AutoConfig.from_pretrained(args.output_dir)
  config.CPG = False
  args.model_type = config.model_type
  #tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
  tokenizer = AutoTokenizer.from_pretrained(args.output_dir)

  logger.info("loading from existing model {}".format(args.model_name_or_path))
  #pdb.set_trace()
  # model = AutoModelForTokenClassification.from_pretrained(
  #       args.model_name_or_path,
  #       config=config,
  #       cache_dir=args.cache_dir,
  #   )
  model = AutoModelForTokenClassification.from_pretrained(args.output_dir,config=config)
  #pdb.set_trace()
  lang2id = LANG2ID if args.l2v else None
  logger.info("Using lang2id = {}".format(lang2id))

  # Make sure only the first process in distributed training loads model/vocab
  if args.local_rank == 0:
    torch.distributed.barrier()
  model.to(args.device)
  # tokenizer = AutoTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case, use_fast=False)

  #pdb.set_trace()
  logger.info("Training/evaluation parameters %s", args)

  # Initialization for evaluation
  results = {}
  
  best_checkpoint = args.output_dir
  best_f1 = 0

  logger.info("Loading the best checkpoint from {}\n".format(best_checkpoint))
  
  load_lang_adapter = args.predict_lang_adapter
  model.model_name = args.model_name_or_path
  task_name="ner"
  load_adapter = best_checkpoint + "/" + task_name
  print(load_adapter)
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
  for language in languages:
    print(language)
    lang_adapter_name = model.load_adapter("LAs/"+language)
    lang_adapter_names.append(lang_adapter_name)

  fusion_path_ = "/".join(load_adapter.split("/")[:-1])+"/"+",".join(lang_adapter_names)
  model.load_adapter_fusion(fusion_path_)
  model.load_adapter(load_adapter)
  model.set_active_adapters([lang_adapter_names, task_name])
  adap_ids = torch.tensor([LANG2ID[it] for it in lang_adapter_names])
  model.to(args.device)
  output_test_results_file = os.path.join(args.output_dir, "test_results.txt")
  with open(output_test_results_file, "a") as result_writer:
    for lang in args.predict_langs.split(','):
      if not os.path.exists(os.path.join(args.data_dir, lang, 'test.{}'.format(args.model_name_or_path))):
        logger.info("Language {} does not exist".format(lang))
        continue
      adapter_weight = None
      if not args.adapter_weight:
        if (adapter_args.train_adapter or args.test_adapter) and not args.adapter_weight:
          pass
    
      else:
        if args.adapter_weight != "0":
            adapter_weight = [float(w) for w in args.adapter_weight.split(",")]
        pdb.set_trace()
        model.set_active_adapters([lang_adapter_names, [task_name]])
      result, predictions = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="test", lang=lang, adap_ids=adap_ids, lang2id=lang2id, adapter_weight=adapter_weight, lang_adapter_names=lang_adapter_names, task_name=task_name, calc_weight_step=args.calc_weight_step)

      # Save results
      if args.predict_save_prefix is not None:
        result_writer.write("=====================\nlanguage={}_{}\n".format(args.predict_save_prefix, lang))
      else:
        result_writer.write("=====================\nlanguage={}\n".format(lang))
      for key in sorted(result.keys()):
        result_writer.write("{} = {}\n".format(key, str(result[key])))
      # Save predictions
      if args.predict_save_prefix is not None:
        output_test_predictions_file = os.path.join(args.output_dir, "test_{}_{}_predictions.txt".format(args.predict_save_prefix, lang))
      else:
        output_test_predictions_file = os.path.join(args.output_dir, "test_{}_predictions.txt".format(lang))
      infile = os.path.join(args.data_dir, lang, "test.{}".format(args.model_name_or_path))
      idxfile = infile + '.idx'
      save_predictions(args, predictions, output_test_predictions_file, infile, idxfile)

def save_predictions(args, predictions, output_file, text_file, idx_file, output_word_prediction=True):
  # Save predictions
  with open(text_file, "r") as text_reader, open(idx_file, "r") as idx_reader:
    text = text_reader.readlines()
    index = idx_reader.readlines()
    assert len(text) == len(index)

  # Sanity check on the predictions
  with open(output_file, "w") as writer:
    example_id = 0
    prev_id = int(index[0])
    for line, idx in zip(text, index):
      if line == "" or line == "\n":
        example_id += 1
      else:
        cur_id = int(idx)
        output_line = '\n' if cur_id != prev_id else ''
        if output_word_prediction:
          output_line += line.split()[0] + '\t'
          output_line += line.split()[1] + '\t'
          #pdb.set_trace()
        output_line += predictions[example_id].pop(0) + '\n'
        writer.write(output_line)
        prev_id = cur_id

if __name__ == "__main__":
  main()
