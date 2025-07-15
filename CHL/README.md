## 🧬 Computational Health Laboratory (University of Pisa, 03/06/2022)

This folder contains my final project for the *Computational Health Laboratory* course at the University of Pisa.  
The project investigates machine learning approaches for biomarker extraction and neurodegenerative disease classification using gene expression data.

---

### 📄 Final Project: Machine Learning for Neurodegenerative Disease Diagnosis and Biomarker Extraction

- **Goal:**  
  Apply machine learning methods to identify predictive biomarkers and build diagnostic models for six neurodegenerative diseases, using high-dimensional gene expression datasets.

- **Diseases analyzed:**  
  - Alzheimer’s Disease (AD)  
  - Huntington’s Disease (HD)  
  - Parkinson’s Disease (PD)  
  - Amyotrophic Lateral Sclerosis (ALS)  
  - Multiple Sclerosis (MS)  
  - Schizophrenia (SCHIZ)

---

### 🧪 Key methods and pipeline

- **Data:**  
  - Publicly available microarray dataset (GSE26927, Gene Expression Omnibus) with 118 post-mortem brain tissue samples.

- **Analysis steps:**  
  ✅ Exploratory data analysis (EDA)  
  ✅ Differential gene expression analysis (limma package, R)  
  ✅ Feature selection (logistic regression with L1 penalty, random forest importance)  
  ✅ Predictive modeling (binary and multi-class classifiers, scikit-learn, Python)  
  ✅ Dimensionality reduction and visualization (UMAP, t-SNE, PCA)  
  ✅ Clustering analysis (k-means, hierarchical clustering, factoextra package, R)  
  ✅ Pathway enrichment analysis (g:Profiler, Cytoscape, EnrichmentMap)

---

### 🔬 Highlights and results

- Identified 21 key genes with strong predictive power across diseases.  
- Achieved **perfect test classification scores** for HD, ALS, and SCHIZ binary classifiers.  
- Developed a multi-class predictor based on selected genes, achieving high performance.  
- Conducted pathway enrichment and protein-protein interaction analyses, linking gene biomarkers to biological mechanisms and clinical phenotypes.

---

### 🏆 Outcome

Final grade: **30/30**

---

### 💡 Key learning points

- Experience combining R and Python for biomedical data science  
- Practical exposure to differential gene expression workflows and feature engineering  
- Integration of machine learning, clustering, and pathway enrichment analyses  
- Interpretation of complex omics data for translational and clinical insights

---

### 📂 Structure
/CHL \
├── CHL_Report.pdf \
├── CHL_2022_neurodegenerative.Rproj \
├── neurodegenerative_analysis.R \
└── README.md
