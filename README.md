*A Repository for ML univerisitarian courseworks & project that’s equal parts supervised learning and supervised chaos.*

---

### 📍 Objective
The goal of this project is to develop a foundational understanding of **supervised learning algorithms** by implementing, analyzing, and comparing multiple models on a real-world dataset.

The chosen task is a **binary classification problem** — predicting diabetes outcomes based on medical and demographic data. 

The dataset contains over **1,000 data points**, derived and expanded from the **[Diabetes Dataset on Kaggle](https://www.kaggle.com/datasets/saurabh00007/diabetescsv)**.

---

### 📍 Project Structure

#### ⤿ Part 1 — Data Selection

- **Dataset:** Diabetes Dataset (Kaggle, see link above)  
- **Task:** Binary classification (Diabetic / Non-Diabetic)  
- **Number of samples:** 1,200 (originally 769, expanded for analysis)  

---

#### ⤿ Part 2 — Data Preprocessing

Steps include:

- Handling missing or noisy data  
- Normalizing or standardizing continuous features  
- Splitting data into **training**, **validation**, and **test** sets  
- Optional dimensionality reduction (e.g., PCA) or feature selection  

---

#### ⤿ Part 3 — Model Implementation and Training
The following supervised learning models are implemented using **Python** and **scikit-learn**:

| Model | Type | Notes |
|-------|------|-------|
| Gaussian Naïve Bayes | Classification | Simple probabilistic baseline |
| Logistic Regression | Binary classification | Core linear model for probabilities |
| Decision Tree | Classification | Interpretable model, useful for feature insights |
| Random Forest | Classification | Ensemble approach for improved stability |
| SVM (Linear & RBF Kernel) | Binary classification | Margin-based classifier for complex boundaries |

*(Softmax and Linear Regression excluded as this is a binary classification task.)*

---

#### ⤿ Part 4 — Evaluation
Models are compared using:
- **Accuracy**, **Precision**, **Recall**, **F1-Score**  
- **Confusion Matrix**  
- **ROC Curve** and **AUC**  
- **Training vs. Validation** performance  
- *(Optional)* computational cost and training time  

---

#### ⤿ Part 5 — Comparative Analysis
A technical report (PDF, prepared in **LaTeX**) includes:
- Model performance comparison and interpretation  
- Influence of model assumptions  
- Observations on overfitting and generalization  
- Visualizations: learning curves, decision boundaries, feature importance  

---

#### ⤿ Deliverables
1. **Technical Report (PDF)** — includes introduction, methodology, results, and conclusion.  
2. **Google Colab Notebook** — clean, well-commented, and reproducible experiments.  

---

#### ⤿ Tools & Libraries
- **Python 3.x**  
- **NumPy**, **Pandas**  
- **Matplotlib**, **Seaborn**  
- **Scikit-learn**  
- **Google Colab**  
- **LaTeX** (for the report)

---

##### 🔖 License

Developed as part of an academic requirement. Licensing details TBD, but Apache 2.0 just to be on the safer side.

---
  
#### 👥 Author(s)

[**Gyanluca**]   ╰┈➤ˎˊ˗ (https://github.com/gyanluca)  
Me — *still debugging life, one dataset at a time.*  
  
[![GitHub - Alx-a-cod, Author](https://img.shields.io/badge/author-Alx--a--cod-F2928D?logo=github)](https://github.com/Alx-a-cod)  [![GitHub - Gyanluca, Author](https://img.shields.io/badge/author-Gyanluca-4a74c2?logo=github)](https://github.com/gyanluca)
