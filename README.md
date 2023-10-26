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
TBD

### 4. Trained model checkpoint
[link](https://drive.google.com/drive/folders/1ihkwheV6x2tKEPAxRoczqATIiXjJ7PDY?usp=sharing)

### Cite
@misc{rathore2023zgul,
      title={ZGUL: Zero-shot Generalization to Unseen Languages using Multi-source Ensembling of Language Adapters}, 
      author={Vipul Rathore and Rajdeep Dhingra and Parag Singla and Mausam},
      year={2023},
      eprint={2310.16393},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
