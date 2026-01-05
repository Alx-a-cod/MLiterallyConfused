# Machine Learning Course Repository

*A Repository for ML university courseworks & projects that's equal parts supervised learning and supervised chaos.*

---

## Repository Contents

This repository contains two major machine learning projects completed as part of a Master's degree in Computer Science and Artificial Intelligence:

1. **Part 1-4: Diabetes Classification Project** — Binary classification using traditional supervised learning models
2. **Part 5: Dimensionality Reduction Comparative Analysis** — Comprehensive evaluation of PCA vs. Autoencoders across three ML domains

---

## ▸ Project 1: Diabetes Classification (Parts 1-4)

### 📍 Objective
Develop a foundational understanding of **supervised learning algorithms** by implementing, analyzing, and comparing multiple models on a real-world dataset.

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

## ▸ Project 2: Dimensionality Reduction Analysis (Part 5)

### 📍 Objective
Conduct a **systematic comparative analysis** of linear (PCA) versus nonlinear (Autoencoder) dimensionality reduction techniques across three diverse machine learning domains: **image classification**, **regression**, and **clustering**.

This project rigorously evaluates performance-efficiency trade-offs, reconstruction quality, and task-specific efficacy to establish evidence-based guidelines for dimensionality reduction method selection.

---

### 📍 Datasets & Tasks

| Dataset | Task Type | Original Dimensions | Reduced Dimensions | Samples |
|---------|-----------|--------------------|--------------------|---------|
| **Fashion-MNIST** | Image Classification | 784 (28×28 pixels) | 50 | 70,000 |
| **California Housing** | Regression | 8 features | 5 features | 20,433 |
| **Credit Card Customers** | Clustering (K-Means) | 17 features | 8 features | 8,950 |

---

### 📍 Methodology

#### Dimensionality Reduction Methods
1. **Principal Component Analysis (PCA)**
   - Linear dimensionality reduction via eigendecomposition
   - Variance-maximizing orthogonal projections
   - Interpretable component loadings

2. **Autoencoders (AE)**
   - Nonlinear dimensionality reduction via neural networks
   - Architecture: Encoder → Bottleneck → Decoder
   - Trained with MSE reconstruction loss

#### Model Architectures

**Fashion-MNIST Classification:**
- **CNN (Original)**: 2×Conv2D → MaxPool → Dense layers (225,930 parameters)
- **Dense (PCA/AE)**: Fully connected network on 50D features (17,962 parameters)

**California Housing Regression:**
- **FNN (Original)**: 4-layer feedforward network on 8D features
- **FNN (PCA/AE)**: Same architecture on 5D reduced features

**Credit Card Clustering:**
- **K-Means**: Applied to original 17D, PCA 8D, and AE 8D representations
- Optimal k=3 determined via elbow method + silhouette analysis

---

### 📍 Key Findings

#### Fashion-MNIST (Image Classification)
- **CNN Original**: 92.74% accuracy, 225,930 parameters
- **Dense + PCA**: 88.10% accuracy, 92% parameter reduction, 3× training speedup
- **Dense + AE**: 85.92% accuracy, 30.91% better reconstruction than PCA
- **Verdict**: CNN exploits spatial structure best; PCA offers strong efficiency-accuracy trade-off for edge deployment

#### California Housing (Regression)
- **Original (8D)**: R²=0.797
- **PCA (5D)**: R²=0.682 (−11.5 pp), minimal computational gains
- **AE (5D)**: R²=0.647 (−15.0 pp)
- **Verdict**: Dimensionality reduction counterproductive for low-dimensional linear data (d<20)

#### Credit Card (Clustering)
- **K-Means Original**: Silhouette=0.255, 22 iterations
- **K-Means + PCA**: Silhouette=0.277 (+8.6%)
- **K-Means + AE**: Silhouette=0.357 (+40.0%), 9 iterations, 52.9% memory reduction
- **Verdict**: Autoencoder creates well-separated spherical clusters, simultaneously improving quality AND efficiency

---

### 📍 Decision Framework

**When to use PCA:**
- Linear relationships in data (e.g., tabular socioeconomic data)
- Interpretability required (loading vectors enable domain expert validation)
- High dimensionality (d > 50) with computational constraints
- Deterministic, training-free method needed

**When to use Autoencoders:**
- Nonlinear manifolds (images, behavioral data)
- Reconstruction quality prioritized
- Unsupervised tasks (clustering, anomaly detection)
- End-to-end integration with deep learning pipelines

**When to avoid dimensionality reduction:**
- Low original dimensionality (d < 20)
- Maximum accuracy required (safety-critical applications)
- Sufficient computational resources available

---

### 📍 Computational Efficiency Results

| Task | Metric | Original | PCA | Autoencoder |
|------|--------|----------|-----|-------------|
| **Fashion-MNIST** | Parameters | 225,930 | 17,962 (−92%) | 17,962 (−92%) |
|  | Training time | 300s | 100s (3×) | 100s (3×) |
|  | Inference | 0.38ms | 0.13ms (2.9×) | 0.15ms (2.5×) |
| **Housing** | Parameters | 3,585 | 3,393 (−5.4%) | 3,393 (−5.4%) |
|  | Training time | 50.0s | 35.2s (1.42×) | 40.0s (1.25×) |
| **Credit Card** | K-Means iterations | 22 | 25 | 9 (2.4× faster) |
|  | Memory | 0.40 KB | 0.19 KB (−52.9%) | 0.19 KB (−52.9%) |

---

### 📍 Visualizations & Analysis

The project includes 19 comprehensive visualizations:

**Performance Metrics:**
- Classification accuracy/precision/F1 comparisons
- Confusion matrices across models
- ROC curves and AUC analysis
- Regression residual plots and predictions scatter

**Dimensionality Reduction Quality:**
- PCA variance explained (scree plots, cumulative variance)
- PCA component loadings heatmaps
- Autoencoder training curves
- Original vs. PCA vs. AE reconstruction comparisons

**Clustering Analysis:**
- Elbow method curves (k=2 to k=10)
- Silhouette score analysis
- 2D cluster visualizations with centroids

**Computational Cost:**
- Parameter count comparisons
- Training/inference time analysis
- Model size and memory footprint

---

### 📍 Technical Report

A comprehensive **40-page LaTeX technical report** includes:

1. **Impact of Model Assumptions on Performance**
   - CNN spatial structure exploitation
   - PCA linearity assumptions
   - K-Means spherical cluster assumptions

2. **Dimensionality Reduction: PCA vs. Autoencoder**
   - Reconstruction quality analysis
   - Task-dependent efficacy guidelines
   - When each method excels

3. **Computational Efficiency vs. Accuracy Trade-offs**
   - Pareto frontier analysis
   - Deployment scenario recommendations

4. **Overfitting Analysis and Regularization**
   - Learning curve analysis
   - Regularization techniques (dropout, batch normalization, early stopping)

5. **Feature Importance and Model Interpretability**
   - PCA component interpretation
   - Confusion matrix error pattern analysis
   - Cluster semantic analysis

6. **Synthesis and Recommendations**
   - Decision framework table
   - Task-specific recommendations
   - Methodological insights
   - Future work directions

---

## ⤿ Deliverables

### Project 1 (Parts 1-4):
1. **Technical Report (PDF)** — Introduction, methodology, results, and conclusion  
2. **Google Colab Notebook** — Clean, well-commented, and reproducible experiments  

### Project 2 (Part 5):
1. **Technical Report (PDF)** — 40-page comprehensive comparative analysis with 19 figures
2. **Jupyter Notebook** — Complete implementation with all three datasets
3. **19 PNG Visualizations** — High-quality figures for all analyses

---

## ⤿ Tools & Libraries

### Core ML Stack:
- **Python 3.x**  
- **NumPy**, **Pandas**  
- **Scikit-learn**  

### Deep Learning:
- **TensorFlow / Keras** (for autoencoders and CNNs)
- **PyTorch** (alternative implementations)

### Visualization:
- **Matplotlib**, **Seaborn**  
- **Plotly** (interactive visualizations)

### Development Environment:
- **Google Colab** (GPU acceleration for neural networks)
- **Jupyter Notebook**

### Documentation:
- **LaTeX** (technical reports with professional typesetting)
- **Overleaf** (collaborative LaTeX editing)

---

## ➥ Key Learning Outcomes

### From Project 1:
- Implementation of classical supervised learning algorithms
- Understanding bias-variance tradeoffs
- Model evaluation and validation techniques
- Ensemble methods and their advantages

### From Project 2:
- Deep understanding of dimensionality reduction theory and practice
- Trade-offs between linear and nonlinear methods
- Performance-efficiency optimization for different deployment scenarios
- Rigorous experimental methodology and statistical analysis
- Technical writing and visualization for ML research

---

## 🔖 License

Developed as part of an academic requirement at a Master's level Computer Science and Artificial Intelligence program.

Licensed under **Apache 2.0** for safety and open collaboration.

---

## 📧 Contact

For questions, collaboration, or academic inquiries, feel free to reach out through GitHub issues or repository discussions.

---

*"In theory, theory and practice are the same. In practice, they are not."* — Attributed to various computer scientists, perfectly describing ML coursework.

  
#### 👥 Author(s)

[**Gyanluca**]   ╰┈➤ˎˊ˗ (https://github.com/gyanluca)  
Me — *:)*  
  
[![GitHub - Alx-a-cod, Author](https://img.shields.io/badge/author-Alx--a--cod-F2928D?logo=github)](https://github.com/Alx-a-cod)  [![GitHub - Gyanluca, Author](https://img.shields.io/badge/author-Gyanluca-4a74c2?logo=github)](https://github.com/gyanluca)
