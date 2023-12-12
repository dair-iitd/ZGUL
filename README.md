## Official Code for "ZGUL: Zero-shot Generalization to Unseen Languages using Multi-source Ensembling of Language Adapters"

### 1. Environment Setup
TBD

### 2. Zero-shot Inference
EM Steps (T) and LR (lr) for each target language (tuned on closest language dev set) along with Test F1 scores can be found in em_params_zero_shot.png . Please run as follows:

First copy infer* files from `scripts' directory to current one

* Germanic
```
bash infer_germanic.sh
```
* Slavic
```
bash infer_slavic.sh
```
* African
```
bash infer_african.sh
```
* Indic
```
bash infer_indic.sh
```
### 3. Training Instructions
First copy train* files from `scripts' directory to current one

* Germanic
```
bash train_udpos.sh en,is,de en,is,de
```
* Slavic
```
bash train_udpos.sh en,ru,cs en,ru,cs
```
* African
```
bash train_masa.sh en,amh,swa,wol en_conll,am,sw,wo
```
* Indic
```
bash train_panx.sh en,hi,bn,ur en_ner,hi,bn,ur
```

### 4. Trained model checkpoint
[link](https://drive.google.com/drive/folders/1ihkwheV6x2tKEPAxRoczqATIiXjJ7PDY?usp=sharing)

### Cite
The codebase is a part of the work [ZGUL: Zero-shot Generalization to Unseen Languages using Multi-source Ensembling of Language Adapters](https://aclanthology.org/2023.emnlp-main.431/). If you use or extend our work, please cite the following paper:
```
@inproceedings{rathore-etal-2023-zgul,
    title = "{ZGUL}: Zero-shot Generalization to Unseen Languages using Multi-source Ensembling of Language Adapters",
    author = "Rathore, Vipul  and
      Dhingra, Rajdeep  and
      Singla, Parag  and
      {Mausam}",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.431",
    pages = "6969--6987",
}

```

### Acknowledgements
Our codebase is built upon [Adapterhub's](https://arxiv.org/abs/2007.07779). For more details on the transformers source code used, we refer the user to their [repository](https://github.com/adapter-hub/adapter-transformers/tree/master/src/transformers).

For more details on the dataset and training scripts used, we refer the user to [Google xtreme](https://github.com/google-research/xtreme) repo.
