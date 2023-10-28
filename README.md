## Official Code for "ZGUL: Zero-shot Generalization to Unseen Languages using Multi-source Ensembling of Language Adapters"

### 1. Environment Setup
TBD

### 2. ZGUL Inference
EM Steps (T) and LR (lr) for various languages (tuned on closest language dev set) along with Test F1 scores for each target -  

| Target lang | Closest source lang | T | lr | Target F1 (test) | 
| :------------ |:---------------:| -----:| -----:| -----:|
| Fo      | Is | 1 | 0.5  | 76.9 |
| Got     | Is | 1 | 0.5  | 20.2 |  
| Gsw     | De | 10 | 0.05 |  64.8 |
| Germanic Avg. |   |   |   |  **54**  |
| Qpm     | Ru | 5 | 0.05 | 50.1 |
| Hsb     | Ru | 5 | 0.05 | 77.2 |
| Cu     | Ru | 5 | 0.05 | 34.1 |
| Orv      | Ru | 5 | 0.05 | 63 |
| Slavic Avg. |   |   |   |  **56.1**  |
| Hau     | Wo | 5 | 1.0 | 53.6 |
| Ibo     | Sw | 10 | 1.0 | 56.8 |
| Kin     | Sw | 10 | 1.0 | 56.2 |
| Lug      | Sw | 10 | 1.0 | 54.2 |
| Luo      | Wo | 5 | 1.0 | 40.2 |
| Pcm      | En | 10 | 0.5 | 66.5 |
| African Avg. |   |   |   |   **54.6** |
| As      | Bn | 5 | 0.1 | 74.4 |
| Bh      | Hi | 5 | 0.1 | 64.1 |
| Indic Avg. |   |   |   |   **69.3** |

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
The codebase is a part of the work [ZGUL: Zero-shot Generalization to Unseen Languages using Multi-source Ensembling of Language Adapters](https://arxiv.org/abs/2310.16393). If you use or extend our work, please cite the following paper:
```
@inproceedings{rathore2023zgul,
  title={ZGUL: Zero-shot Generalization to Unseen Languages using Multi-source Ensembling of Language Adapters},
  author={Rathore, Vipul and Dhingra, Rajdeep and Singla, Parag and Mausam},
  booktitle={Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, {EMNLP} 2023 (Volume 1: Long Papers)},
  year={2023}
}
```

### Acknowledgements
Our codebase is built upon [Adapterhub's](https://arxiv.org/abs/2007.07779). For more details on the transformers source code used, we refer the user to their [repository](https://github.com/adapter-hub/adapter-transformers/tree/master/src/transformers).

For more details on the dataset and training scripts used, we refer the user to [Google xtreme](https://github.com/google-research/xtreme) repo.
