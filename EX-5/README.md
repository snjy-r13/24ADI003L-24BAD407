
---

# 🔬 Scenario 1: Breast Cancer Diagnosis using KNN

## 📌 Objective

Build a binary classification model to predict whether a tumor is **Malignant (M)** or **Benign (B)** using selected numerical features from a breast cancer dataset.

## 🧠 Algorithm Used

- **K-Nearest Neighbors (KNN)**
- Distance-based supervised learning algorithm
- Hyperparameter tuning for optimal **K**

## ⚙️ Workflow

### 1. Data Loading
- Dataset loaded from `breast-cancer.csv`

### 2. Feature Selection
Selected features:
- `radius_mean`
- `texture_mean`
- `perimeter_mean`
- `area_mean`
- `smoothness_mean`

### 3. Preprocessing
- Label Encoding: `M → 1`, `B → 0`
- Train/Test Split (80/20)
- Feature Scaling using `StandardScaler`

### 4. Model Training
- K values tested from **1 to 20**
- Accuracy plotted against K
- Best K selected using maximum validation accuracy

### 5. Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

### 6. Advanced Analysis
- Misclassified case inspection
- Decision boundary visualization (first two features)
- Interpretation of classification regions

## 📊 Outputs
- Accuracy vs K graph
- Confusion matrix heatmap
- Decision boundary plot
- Misclassification report

---

# 🏦 Scenario 2: Loan Approval Prediction using Decision Tree

## 📌 Objective

Predict whether a loan application will be **Approved** or **Rejected** based on applicant demographic and financial information.

## 🧠 Algorithm Used

- **Decision Tree Classifier**
- Comparison between:
  - Shallow Tree (max_depth=3)
  - Deep Tree (unrestricted)

## ⚙️ Workflow

### 1. Data Loading
- Dataset loaded from `loan-predication.csv`

### 2. Data Preprocessing

#### Missing Value Handling
- Categorical columns → filled with mode
- Numerical columns → filled with median

#### Encoding
- Label encoding for categorical variables
- Dropped `Loan_ID` (non-predictive)

### 3. Train/Test Split
- 80/20 split
- Random state fixed for reproducibility

### 4. Model Training
- Default Decision Tree
- Shallow Tree (controlled depth)
- Deep Tree (fully grown)

### 5. Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- Classification Report
- Confusion Matrices

### 6. Model Behavior Analysis
- Overfitting detection via train vs test accuracy
- Feature importance ranking
- Tree structure visualization

## 📊 Outputs
- Confusion matrix comparison (shallow vs deep)
- Decision tree visualization
- Feature importance chart
- Overfitting detection summary
