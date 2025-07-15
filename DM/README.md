## 📊 Data Mining (University of Pisa, A.Y. 19/01/2021)

This folder contains my final project for the *Data Mining* course at the University of Pisa, conducted in collaboration with Diletta Goglia.  
The project analyzed the **customer_supermarket** dataset to explore customer purchasing behavior through data understanding, clustering, classification, and sequential pattern mining.

---

### 📄 Final Project: Analysis of the “customer_supermarket” Dataset

- **Goal:**  
  Understand and model customer purchasing patterns, clean and prepare transactional data, and apply multiple data mining techniques for segmentation, prediction, and pattern discovery.

---

### 🛠️ Main tasks

- **Task 1 — Data Understanding and Preparation:**  
  ✅ Dataset cleaning (duplicates, missing values, outliers, canceled purchases)  
  ✅ Feature engineering (per-customer metrics, entropy measures)  
  ✅ Data visualization (temporal trends, correlations)

- **Task 2 — Data Clustering:**  
  ✅ K-Means clustering (elbow, silhouette, Ward linkage)  
  ✅ DBSCAN clustering (parameter tuning, density analysis)  
  ✅ Hierarchical clustering (dendrogram cuts, linkage comparisons)  
  ✅ Cluster characterization (spending profiles, country patterns, favorite days/months)

- **Task 3 — Data Classification:**  
  ✅ Label creation (based on average spending: low, medium, high)  
  ✅ Model comparison: KNN, SVM, Decision Trees, MLP, Naive Bayes, Random Forest, Voting Classifier  
  ✅ Evaluation: accuracy, precision, recall, confusion matrices, decision boundaries, ROC curves

- **Task 4 — Sequential Pattern Mining:**  
  ✅ Applied Apriori algorithm to extract frequent product sequences  
  ✅ Custom Python implementation (`gsp.py`) to identify and analyze recurring customer purchase patterns

---

### 🔬 Key results

- Identified high- and low-spending customer groups with clear behavioral profiles  
- Achieved top classification performance with Decision Tree, MLP, and Voting classifiers  
- Extracted meaningful frequent product sequences suggesting habitual or wholesale purchase patterns

---

### 🏆 Outcome

Final grade: **30/30**

---

### 💡 Key learning points

- Hands-on experience with advanced data cleaning, feature engineering, and clustering  
- Practical exposure to machine learning model selection and evaluation  
- Development of a custom Python implementation for sequential pattern mining  
- Critical interpretation of clustering and classification results

---

### 📂 Structure
/DM \
├── DM_04_TASK1/ \
│ ├── dataset/ \
│ ├── Task_1.ipynb \
│ └── README.md \
├── DM_04_TASK2/ \
│ ├── dataset/ \
│ ├── Task_2.ipynb \
│ └── README.md \
├── DM_04_TASK3/ \
│ ├── dataset/ \
│ ├── Task_3.ipynb \
│ └── README.md \
├── DM_04_TASK4/ \
│ ├── dataset/ \
│ ├── gsp.py \
│ ├── Task_4.ipynb \
│ └── README.md \
├── DM_Report_04.pdf \
├── DM-project-slides.pdf \
└── README.md 
