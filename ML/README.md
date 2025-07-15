## 📊 Machine Learning (University of Pisa, 10/02/2021)

This folder contains the final project report for the *Machine Learning* course at the University of Pisa.  
The project is a comparative study of several regression models, focusing on model selection, hyperparameter tuning, and experimental evaluation on the CUP dataset.

---

### 📄 Final Project: A Comparison Among Regression Models

- **Goal:**  
  Compare the performance of different regression models, including neural networks, support vector machines, and k-nearest neighbors, using systematic hyperparameter tuning and validation strategies.

- **Models compared:**  
  ✅ Neural Network (Keras)  
  ✅ Support Vector Regressor (Scikit-learn)  
  ✅ K-Neighbors Regressor (Scikit-learn)  
  ✅ Extra Trees Regressor  
  ✅ Decision Tree Regressor

---

### ⚙️ Methodology

- Neural network implemented in Keras, using:
  - Two dense hidden layers (17 units, sigmoid and tanh activations)  
  - Stochastic Gradient Descent (mini-batch)  
  - Custom Mean Euclidean Error (MEE) as evaluation metric

- Model selection approach:
  - Screening phase to narrow hyperparameter ranges  
  - Grid search with 3-fold cross-validation  
  - Internal test set via hold-out split (75% training, 25% test)

- K-Neighbors Regressor with min-max normalization  
- Detailed hyperparameter exploration for all models

---

### 🔬 Key experiments and results

- Best performing models:
  - K-Neighbors Regressor (r²: 0.952/0.941, MEE ≈ 2.88)  
  - RegressorChain SVR with RBF kernel

- Observed:
  - Effect of feature suppression on performance (worse after removing correlated features)  
  - Influence of learning rate, momentum, batch size, and hidden units on neural network learning curves  
  - Detailed performance tables and learning curves included

- Additional experiments:
  - Performance on MONK’s datasets for classification tasks

---

### 🏆 Outcome

Final grade: **29/30**

---

### 💡 Key learning points

- Hands-on experience with multiple ML frameworks (Keras, Scikit-learn)  
- Practical understanding of hyperparameter tuning, grid search, and cross-validation  
- Deepened ability to interpret learning curves and generalization behaviors  
- Developed critical analysis of comparative model performance
