# Hybrid Knowledge Distillation + Selective Distillation for Efficient Retrieval
This repository contains the implementation of a **Hybrid Knowledge Distillation (KD)** framework combined with **Selective Distillation (SD)** for training a lightweight bi-encoder retrieval model.  
The goal is to distill rich semantic similarity signals from a large teacher model into a compact student model while using only the most reliable supervision signals.

---

## 🚀 Overview

Retrieval models benefit significantly from large transformer-based encoders, but such models are computationally expensive.  
This project proposes a **hybrid distillation strategy** that transfers knowledge from a powerful teacher model into a compact student encoder using:

### ✔ Hybrid Distillation Loss  
- **Contrastive Loss** → teaches *relative similarity*  
- **Regression Loss** → aligns *absolute similarity values*  
- Combined to stabilize calibration + ranking performance

### ✔ Selective Distillation  
Only teacher-labeled pairs with **high-confidence similarity scores** (top 20%) are used.  
This reduces noise and ensures the student learns only from high-quality supervision.

---

## 🧠 Model Architecture

### Teacher Model
- `sentence-transformers/multi-qa-mpnet-base-dot-v1`
- Provides high-quality embeddings
- Supplies soft similarity labels for student training

### Student Model
- `microsoft/MiniLM-L12-H384-uncased` backbone
- + a **256-dim projection head**
- + L2-normalized embeddings for contrastive learning

---

## 📊 Dataset

We use the **QReCC conversational retrieval dataset**:

- 63,501 training samples  
- After preprocessing → **61,344 valid (query, doc) pairs**  
- After Selective Distillation → **12,269 high-quality pairs**

Dataset preprocessing includes:
- Query rewriting extraction  
- Passage extraction  
- Cleaning and validation  
- Teacher-score generation  
- Filtering noisy samples

---

## 🔄 Pipeline Workflow

### **1. Data Extraction**
Extract `(query, document)` pairs from QReCC.

### **2. Teacher Soft Label Generation**
Use the teacher model to compute:
- Query embeddings  
- Document embeddings  
- Cosine similarity scores  

Scores saved in:  
`teacher_soft_labels.json`

### **3. Selective Distillation**
Filter top-quality samples based on:
- 80th percentile similarity threshold  
- Final distilled dataset: **12,269 pairs**

### **4. Hybrid Distillation Training**
The student learns using:
- Contrastive similarity matrix  
- Regression target (teacher score)  
- Temperature scaling  
- AdamW + linear warmup  

Model saved to:  
`student_model_sd/`

### **5. Final Evaluation**
Metrics used:
- Pearson correlation  
- MAE (Mean Absolute Error)  
- Recall@K  
- NDCG@K  
- F1-Score  
- Statistical tests (t-test, Wilcoxon, Cohen’s d)

---

## 📈 Key Results

| Metric | Value |
|--------|-------|
| **Pearson Correlation** | 0.3531 |
| **MAE** | 0.1454 |
| **Recall@1** | 1.000 |
| **Recall@10** | 0.400 |
| **NDCG@10** | 0.7347 |
| **F1-Score (binary teacher labels)** | 0.6672 |
| **Effect Size (Cohen’s d)** | –3.623 (large)** |

### Interpretation
- The model accurately captures *high-confidence teacher signals*.  
- Larger errors occur in mid-range similarity (0.7–0.8).  
- Hybrid Distillation + selective filtering improves stability.  
- Low-capacity student model still shows a structural gap from the teacher.
