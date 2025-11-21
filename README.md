# Hybrid Knowledge Distillation + Selective Distillation for Efficient Retrieval

This repository contains the implementation of a **Hybrid Knowledge Distillation (KD)** framework combined with **Selective Distillation (SD)** for training a lightweight bi-encoder retrieval model.

## Overview

Retrieval models benefit significantly from large transformer encoders, but such models are expensive.  
This project distills knowledge from a teacher model into a smaller student encoder using:

- Hybrid loss (contrastive + regression)
- Selective distillation (top 20% high-confidence samples)
- Projection head for stable embedding alignment

## Workflow

1. Data extraction from QReCC
2. Teacher embedding + soft label generation
3. Selective Distillation (80th percentile filtering)
4. Hybrid KD training
5. Evaluation + statistical analysis

## Key Results

- Pearson Correlation: 0.3531  
- MAE: 0.1454  
- Recall@1: 1.0  
- Recall@10: 0.40  
- NDCG@10: 0.7347  
- F1 Score: 0.6672  

## Files

- extracted_pairs.json
- teacher_soft_labels.json
- selected_qrecc_sd.json
- best_student_model.pt
- evaluation_results.json

## Citation

@article{hinton2015distill,
  title={Distilling the knowledge in a neural network},
  author={Hinton, Geoffrey and Vinyals, Oriol and Dean, Jeff},
  journal={arXiv:1503.02531},
  year={2015}
}

@inproceedings{lupart2025disco,
  author={Lupart, Simon and Aliannejadi, Mohammad and Kanoulas, Evangelos},
  title={DiSCo: LLM Knowledge Distillation for Efficient Sparse Retrieval in Conversational Search},
  booktitle={SIGIR '25},
  pages={9--19},
  year={2025},
  doi={10.1145/3726302.3729966}
}
