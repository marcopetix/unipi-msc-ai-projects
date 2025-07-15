## 🗣️ Human Language Technologies (University of Pisa, 17/05/2023)

This folder contains my final project for the *Human Language Technologies* course at the University of Pisa.  
The project focused on solving the **U.S. Patent Phrase to Phrase Matching** (USPPTM) challenge using advanced NLP techniques, including pretrained transformers, fine-tuning, and prompt-based learning (PET).

---

### 📄 Final Project: Conditional Semantic Similarity with Fine-Tuning and Prompt Learning

- **Goal:**  
  Build models to estimate semantic similarity between pairs of patent phrases, conditioned on their technical context (CPC code), to support prior art search and patent vetting.

- **Dataset:**  
  36,473 sentence pairs from U.S. patent descriptions, with CPC codes and human-annotated similarity scores (0 to 1).

---

### 🛠️ Main approaches

- **Transformer Fine-tuning (PyTorch Lightning):**  
  - Models: BERT, DistilBERT, ELECTRA, DeBERTa, BERT-for-Patents, PatentSBERTa  
  - Features: anchor-target pairs + CPC context + derived features (SACT, SACST)  
  - Grid search over architectures (linear, attention, LSTM), loss functions, learning rates, stratification strategies

- **Prompt-based Learning (PET):**  
  - Pattern engineering and verbalizers to transform similarity prediction into masked language modeling  
  - Few-shot and many-shot experiments  
  - Regression readout to convert PET logits into continuous similarity scores

---

### 🔬 Key results

- Achieved ~86–88% Pearson correlation on the internal test set  
- Showed PET is competitive, especially under low-resource conditions  
- Identified architecture and hyperparameter interactions (e.g., attention-based readouts outperform linear and LSTM, DeBERTa and ELECTRA outperform BERT-for-Patents in fine-tuning but not always in PET)

---

### 🏆 Outcome

Final grade: **30/30**

---

### 💡 Key learning points

- Hands-on experience with pretrained transformer models (Hugging Face, PyTorch Lightning)  
- Application of prompt-based learning methods (PET) for regression tasks  
- Efficient hyperparameter tuning using Ray Tune  
- Insight into model selection and optimization for domain-specific NLP tasks

