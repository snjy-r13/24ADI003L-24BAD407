# Machine Learning Ensemble Methods — Classification Guide

## 📚 Overview

This directory contains implementations of **Ensemble Learning** techniques for binary and multi-class classification tasks. Five complementary scenarios are included, demonstrating different ensemble strategies:

- **Scenario 1 (EX06-SC1)**: Bagging (Bootstrap Aggregating)
- **Scenario 2 (EX06-SC2)**: Boosting (AdaBoost & Gradient Boosting)
- **Scenario 3 (EX06-SC3)**: Random Forests
- **Scenario 4 (EX06-SC4)**: Stacking (Meta-Learning)
- **Scenario 5 (EX06-SC5)**: Additional Classification Techniques

Each scenario includes data preprocessing, model training, evaluation, and comparative visualizations.

---

## 🎯 Scenarios Overview

### Scenario 1: Bagging (EX06-SC1.ipynb)

**Dataset**: Diabetes (Binary Classification)  
**Baseline**: Decision Tree Classifier  
**Ensemble**: Bagging Classifier (50 estimators)

#### Key Concept:
Bagging reduces variance by training multiple models on bootstrap samples and averaging predictions.

#### Core Implementation:
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

# Baseline
dt = DecisionTreeClassifier(random_state=42, max_depth=6)
dt.fit(X_train, y_train)

# Ensemble
bag = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=42),
    n_estimators=50,           # Number of base models
    max_samples=0.8,           # Fraction of samples per bootstrap
    max_features=0.8,          # Fraction of features per model
    bootstrap=True,            # Use bootstrap sampling
    random_state=42,
    n_jobs=-1                  # Parallel processing
)
bag.fit(X_train, y_train)
```

#### Metrics:
- Accuracy: Decision Tree vs Bagging
- Confusion Matrix visualization
- Classification Report (Precision, Recall, F1)

#### Key Insight:
✅ Bagging typically improves generalization by reducing overfitting  
✅ Works well with high-variance models (deep trees)

---

### Scenario 2: Boosting (EX06-SC2.ipynb)

**Dataset**: Customer Churn (Binary Classification)  
**Methods**: AdaBoost & Gradient Boosting

#### Key Concept:
Boosting sequentially trains models, focusing on misclassified samples. Each new model reduces errors of previous models.

#### Core Implementation:

**AdaBoost**:
```python
from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier(
    n_estimators=100,          # Number of boosting rounds
    learning_rate=0.5,         # Shrinkage parameter
    random_state=42
)
ada.fit(X_train, y_train)
y_pred = ada.predict(X_test)
y_prob = ada.predict_proba(X_test)[:, 1]
```

**Gradient Boosting**:
```python
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(
    n_estimators=100,          # Number of boosting stages
    learning_rate=0.1,         # Shrinkage (smaller = slower, better)
    max_depth=3,               # Depth of trees
    random_state=42
)
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)
y_prob = gb.predict_proba(X_test)[:, 1]
```

#### Evaluation Metrics:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix (Heatmap)
- ROC Curve & AUC Score
- Feature Importance
- Cross-Validation Scores

#### Key Insight:
✅ **AdaBoost**: Simple, fast, good for weak learners  
✅ **Gradient Boosting**: More powerful, better error correction  
⚠️ Both prone to overfitting; monitor learning_rate and n_estimators

---

### Scenario 3: Random Forests (EX06-SC3.ipynb)

**Dataset**: Income Prediction (Binary Classification)  
**Features**: Age, EducationYears, HoursPerWeek, Experience

#### Key Concept:
Random Forests combine bagging with feature randomness. Each tree is trained on a bootstrap sample using a random subset of features.

#### Core Implementation:
```python
from sklearn.ensemble import RandomForestClassifier

# Hyperparameter tuning
tree_range = [1, 5, 10, 20, 30, 50, 75, 100, 150, 200]
train_scores = []
test_scores = []

for n in tree_range:
    rf = RandomForestClassifier(
        n_estimators=n,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    train_scores.append(rf.score(X_train, y_train))
    test_scores.append(rf.score(X_test, y_test))

best_n = tree_range[np.argmax(test_scores)]

# Final model
rf_best = RandomForestClassifier(n_estimators=best_n, random_state=42)
rf_best.fit(X_train, y_train)
```

#### Evaluation:
- Train vs Test accuracy curves
- Optimal number of trees identification
- Confusion Matrix
- Feature Importance ranking

#### Key Insight:
✅ More robust than single decision trees  
✅ Built-in feature importance  
✅ Parallelizable (n_jobs=-1)  
❌ Less interpretable than single trees

---

### Scenario 4: Stacking (EX06-SC4.ipynb)

**Dataset**: Heart Disease Prediction  
**Base Models**: Logistic Regression, SVM, Decision Tree  
**Meta-Learner**: Logistic Regression

#### Key Concept:
Stacking trains multiple diverse base models, then uses their predictions as input to a meta-learner.

#### Core Implementation:
```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

# Feature scaling (required for LR and SVM)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# Base models (diverse algorithms)
lr = LogisticRegression(max_iter=1000, random_state=42)
svm = SVC(kernel='rbf', probability=True, random_state=42)
dt = DecisionTreeClassifier(max_depth=4, random_state=42)

# Meta-learner
meta = LogisticRegression(max_iter=1000, random_state=42)

# Stacking ensemble
stack = StackingClassifier(
    estimators=[
        ('lr', lr),
        ('svm', svm),
        ('dt', dt)
    ],
    final_estimator=meta,
    cv=5  # Cross-validation folds for meta-features
)

stack.fit(X_train_sc, y_train)
```

#### Evaluation Framework:
```python
# Training accuracy, test accuracy, cross-validation
models = {
    'Logistic Regression': lr,
    'SVM': svm,
    'Decision Tree': dt,
    'Stacking Ensemble': stack
}

results = {}
for name, model in models.items():
    model.fit(X_train_sc, y_train)
    train_acc = model.score(X_train_sc, y_train)
    test_acc = model.score(X_test_sc, y_test)
    cv_scores = cross_val_score(model, X_train_sc, y_train, cv=5)
    
    results[name] = {
        'train': train_acc,
        'test': test_acc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
```

#### Key Insight:
✅ Combines strengths of diverse base models  
✅ Reduces bias and variance through diversity  
⚠️ More complex, risk of overfitting meta-learner

---

### Scenario 5: Advanced Classification (EX06-SC5.ipynb)

**Focus**: Extended classification evaluation and additional ensemble techniques

#### Typical Components:
- Multiple preprocessing strategies
- Cross-validation framework
- Comprehensive metrics (Precision, Recall, F1, AUC)
- ROC/AUC analysis
- Model comparison matrix

---

## 🔧 Shared Code Components

### 1. Standard Imports

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, auc, roc_auc_score
)
from sklearn.ensemble import (
    BaggingClassifier, AdaBoostClassifier, 
    GradientBoostingClassifier, RandomForestClassifier,
    StackingClassifier
)
```

### 2. Data Loading & Exploration

```python
# Load dataset
df = pd.read_csv(r"..\data\dataset_name.csv")

# Basic exploration
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(df.head())
print(df.info())
print(df.describe())
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nTarget distribution:\n{df['target_column'].value_counts()}")
```

### 3. Data Preprocessing

```python
# Drop missing values
df.dropna(subset=['critical_column'], inplace=True)

# Encode categorical variables
le = LabelEncoder()
df['categorical_col'] = le.fit_transform(df['categorical_col'])

# Feature selection
feature_names = ['Age', 'EducationYears', 'Income', ...]
X = df[feature_names]
y = df['target_column']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y  # Maintain class distribution
)

# Feature Scaling (if needed for distance-based models)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)
```

### 4. Model Training & Evaluation

```python
# Generic training function
def train_and_evaluate(model, X_train, X_test, y_train, y_test, model_name):
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    
    print(f"\n{model_name}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  CV Mean  : {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    return {
        'model': model,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'y_pred': y_pred,
        'y_prob': y_prob,
        'cm': confusion_matrix(y_test, y_pred)
    }
```

### 5. Visualization Functions

```python
# Confusion Matrix with custom colors
def plot_custom_cm(cm, title, ax):
    color_map = np.empty(cm.shape, dtype=object)
    color_map[0, 0] = "#00FF6E"  # TN (True Negative)
    color_map[1, 1] = "#F200FF"  # TP (True Positive)
    color_map[0, 1] = "#FF0000"  # FP (False Positive)
    color_map[1, 0] = "#FFB800"  # FN (False Negative)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap=ListedColormap(color_map.flatten()),
                ax=ax, cbar=False)
    ax.set_title(title, fontweight='bold')

# ROC Curve
def plot_roc_curves(models_dict, X_test_sc, y_test):
    plt.figure(figsize=(8, 6))
    
    for name, (model, y_prob) in models_dict.items():
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc_score = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} (AUC={auc_score:.3f})', lw=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

# Feature Importance
def plot_feature_importance(model, feature_names, ax, top_n=10):
    importances = model.feature_importances_
    idx = np.argsort(importances)[-top_n:]
    sorted_idx = idx[np.argsort(importances[idx])]
    
    ax.barh([feature_names[i] for i in sorted_idx], importances[sorted_idx])
    ax.set_xlabel('Importance Score')
    ax.set_title(f'Top-{top_n} Feature Importance')

# Model Comparison
def plot_model_comparison(results_dict):
    models = list(results_dict.keys())
    test_accs = [results_dict[m]['accuracy'] for m in models]
    cv_means = [results_dict[m]['cv_mean'] for m in models]
    cv_stds = [results_dict[m]['cv_std'] for m in models]
    
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.bar(x - 0.2, test_accs, 0.4, label='Test Accuracy', alpha=0.8)
    ax.errorbar(x + 0.2, cv_means, yerr=cv_stds, fmt='o-', 
                label='CV Mean ± Std', capsize=5)
    
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Comparison')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
```

### 6. Classification Report Template

```python
# Comprehensive results summary
print("=" * 60)
print(f"MODEL PERFORMANCE SUMMARY")
print("=" * 60)
print(f"\n{'Model':<25} {'Train':<10} {'Test':<10} {'CV Mean':<10}")
print("-" * 60)

for name, metrics in results.items():
    print(f"{name:<25} {metrics['train']:<10.4f} "
          f"{metrics['test']:<10.4f} {metrics['cv_mean']:<10.4f}")

print("\n" + "=" * 60)
print("DETAILED CLASSIFICATION REPORT (Best Model)")
print("=" * 60)
print(classification_report(y_test, best_y_pred, 
                          target_names=['Class 0', 'Class 1']))
```

---

## 📊 Ensemble Methods Comparison

| Method | Bias ↓ | Variance ↓ | Speed | Complexity | Best For |
|--------|--------|-----------|-------|-----------|----------|
| **Bagging** | - | ✅✅✅ | Fast | Low | High-variance models |
| **AdaBoost** | ✅✅✅ | - | Medium | Medium | Weak learners, unbalanced |
| **Gradient Boost** | ✅✅✅ | ✅ | Slow | High | Accuracy on competition |
| **Random Forest** | - | ✅✅ | Fast | Medium | Feature importance, general |
| **Stacking** | ✅✅ | ✅ | Slow | Very High | Combining diverse models |

---

## 💡 When to Use Which Ensemble?

### 🌳 Bagging
✅ Simple, interpretable  
✅ Fast training & prediction  
✅ Works with any base learner  
❌ Limited bias reduction  

**Use When**: Base model has high variance (e.g., deep trees)

### 🎯 Boosting (AdaBoost / Gradient Boosting)
✅ Excellent bias reduction  
✅ Handles imbalanced datasets well  
✅ Feature importance available  
❌ Prone to overfitting, slower  

**Use When**: Need maximum accuracy, can afford training time

### 🌲 Random Forests
✅ Combines bagging + feature randomness  
✅ Fast & parallelizable  
✅ Built-in feature importance  
❌ Less interpretable  

**Use When**: Quick, robust solution with feature insights needed

### 🔗 Stacking
✅ Combines diverse learners effectively  
✅ Reduces both bias and variance  
✅ Often state-of-the-art  
❌ Complex, high overfitting risk  

**Use When**: Combining different model types (LR, SVM, Trees)

---

## 🎓 Workflow Template

```python
# 1. Load & explore
df = pd.read_csv('data.csv')
print(df.describe(), df.isnull().sum())

# 2. Preprocess
X, y = prepare_data(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
X_train_sc, X_test_sc = scale_features(X_train, X_test)

# 3. Train multiple models
results = {}
for model_name, model in models.items():
    results[model_name] = train_and_evaluate(model, X_train_sc, X_test_sc, y_train, y_test)

# 4. Compare and visualize
plot_model_comparison(results)
plot_confusion_matrices(results)
plot_roc_curves(results)

# 5. Analyze best model
best_model = max(results, key=lambda x: results[x]['accuracy'])
print(f"\nBest Model: {best_model}")
print(classification_report(y_test, results[best_model]['y_pred']))
```

---

## 📁 Dataset Examples

| Scenario | Dataset | Size | Task | Target |
|----------|---------|------|------|--------|
| SC1 | Diabetes | ~768 rows | Binary | Diabetes (0/1) |
| SC2 | Customer Churn | ~7043 rows | Binary | Churn (Yes/No) |
| SC3 | Income | ~30K rows | Binary | Income (≤50K / >50K) |
| SC4 | Heart Disease | ~918 rows | Binary | Disease (0/1) |
| SC5 | Variable | Variable | Variable | Variable |

---

## 🔗 Dependencies

```python
pandas              # Data manipulation
numpy               # Numerical computing
matplotlib          # Plotting
seaborn             # Statistical visualization
scikit-learn        # ML algorithms & metrics
```

---

## 📝 Best Practices

1. **Always stratify splits**: `stratify=y` to maintain class distribution
2. **Scale when necessary**: LR, SVM need scaling; tree-based don't
3. **Use cross-validation**: Don't rely on single train/test split
4. **Monitor overfitting**: Compare train vs test accuracy
5. **Start simple**: Baseline → Ensemble → Stacking
6. **Tune hyperparameters**: Use GridSearchCV or RandomizedSearchCV
7. **Handle imbalance**: Use class_weight='balanced' or adjust threshold
8. **Document results**: Save metrics for comparison

---

## 🚀 Advanced Topics

- [ ] Hyperparameter optimization (GridSearchCV, RandomizedSearchCV)
- [ ] Feature engineering & selection
- [ ] Handling imbalanced classes (SMOTE, class_weight)
- [ ] Threshold tuning for ROC curves
- [ ] Ensemble stacking with meta-features
- [ ] Neural networks as base learners
- [ ] Production deployment (pickle, ONNX)

---

**Last Updated**: 2025  
**Author**: Shared Code Repository
