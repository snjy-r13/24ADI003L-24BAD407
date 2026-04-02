# Movie Recommendation System — Collaborative Filtering

## 📚 Overview

This directory contains implementations of **Collaborative Filtering (CF)** approaches for building movie recommendation systems using the MovieLens dataset. Two complementary scenarios are included:

- **Scenario 1 (SC-1)**: User-Based Collaborative Filtering
- **Scenario 2 (SC-2)**: Item-Based Collaborative Filtering

Both scenarios demonstrate different recommendation strategies and include comparative analysis.

---

## 📊 Scenarios

### Scenario 1: User-Based Collaborative Filtering (SC-1.ipynb)

**Method**: KNN with Cosine Similarity  
**Concept**: Find similar users and recommend movies they rated highly.

#### Key Steps:
1. Load MovieLens 100K dataset
2. Build user-item rating matrix
3. Compute cosine similarity between users
4. For each user, find K most similar users
5. Predict ratings using weighted average from neighbors
6. Rank and recommend top-N movies

#### Core Functions:
```python
get_top_n_similar_users(user_id, sim_df, n=5)
# Returns top-N users most similar to target user

predict_ratings(user_id, user_item_df, sim_df, k=10)
# Predicts ratings for unrated movies using KNN weighted average

recommend_movies(user_id, user_item_df, sim_df, movies_meta, k=10, top_n=10)
# Returns DataFrame with top-N recommendations for a user
```

#### Performance Metrics:
- RMSE, MAE on train/test split
- Density: ~1.06% (very sparse matrix)
- Similarity range: User cosine similarity [0.0 to 1.0]

---

### Scenario 2: Item-Based Collaborative Filtering (SC-2.ipynb)

**Methods**: Cosine Similarity & Pearson Correlation  
**Concept**: Find similar movies and aggregate ratings from users who rated similar items.

#### Key Steps:
1. Load MovieLens dataset
2. Build item-user rating matrix (movies × users)
3. Compute item similarity (Cosine + Pearson)
4. For each user, find their rated movies
5. Find similar items to those movies
6. Aggregate ratings, predict and rank

#### Core Functions:
```python
get_similar_items(item_id, sim_df, n=10)
# Returns top-N items most similar to target item

item_based_recommend(user_id, ratings_df, item_user_df, sim_df, 
                     movies_meta, k_neighbors=10, top_n=10)
# Item-based CF recommendation engine

user_based_recommend(user_id, ui_df, sim_df, movies_meta, ...)
# User-based CF (for comparison/baseline)
```

#### Similarity Measures:
- **Cosine Similarity**: Angle between rating vectors (recommended for sparse data)
- **Pearson Correlation**: Correlation coefficient (handles rating biases better)

#### Performance Metrics:
- RMSE/MAE comparison between Item-Based vs User-Based
- Precision@K for different K values
- Similarity distribution analysis

---

## 🔧 Shared Code Components

### 1. Data Loading & Preprocessing

```python
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

DATA_PATH = "./ml-100k/"

# Load ratings, movies, and genres
ratings = pd.read_csv(
    DATA_PATH + "u.data", sep="\t",
    names=["user_id", "movie_id", "rating", "timestamp"]
)

movies_df = pd.read_csv(
    DATA_PATH + "u.item", sep="|", encoding="latin-1",
    names=["movie_id", "title", "release_date", ...]
)

# Data cleaning
ratings["timestamp"] = pd.to_datetime(ratings["timestamp"], unit="s")
ratings = (ratings
    .sort_values("timestamp")
    .drop_duplicates(subset=["user_id", "movie_id"], keep="last")
)
```

### 2. Matrix Construction

```python
# User-Item Matrix (Scenario 1)
user_item_raw = ratings.pivot_table(
    index="user_id", columns="movie_id", values="rating"
)
user_item_zero = user_item_raw.fillna(0)

# Item-User Matrix (Scenario 2)
item_user_raw = ratings.pivot_table(
    index="movie_id", columns="user_id", values="rating"
)
item_user_zero = item_user_raw.fillna(0)

# Sparsity Analysis
n_ratings = ratings.shape[0]
n_users, n_movies = user_item_raw.shape
sparsity = 1 - (n_ratings / (n_users * n_movies))
```

### 3. Similarity Computation

```python
from sklearn.metrics.pairwise import cosine_similarity

# Cosine Similarity (both scenarios)
sim_matrix = cosine_similarity(matrix)
sim_df = pd.DataFrame(sim_matrix, index=matrix.index, columns=matrix.index)

# Pearson Correlation (Scenario 2)
pearson_sim_df = matrix.T.corr(method="pearson")
```

### 4. Train/Test Evaluation

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# Split data
train_df, test_df = train_test_split(ratings, test_size=0.2, random_state=42)

# Rebuild matrices on training data
train_matrix = (train_df
    .pivot_table(index="user_id", columns="movie_id", values="rating")
    .fillna(0)
)

# Evaluate predictions
y_true = test_df['rating'].values
y_pred = predictions  # from recommendation engine

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
```

### 5. Visualization Utilities

```python
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

# Color scheme
BG_WHITE = "#FFFFFF"
ACCENT = "#2C3E50"
ACCENT2 = "#3498DB"
TEAL = "#1ABC9C"
GOLD = "#F39C12"

# Heatmap: User-Item Matrix
plt.figure(figsize=(12, 8))
sns.heatmap(user_item_raw.iloc[:30, :30], 
            cmap="YlGnBu",
            mask=(user_item_raw == 0) | user_item_raw.isna(),
            cbar_kws={"label": "Rating"})
plt.title("User-Item Rating Matrix")

# Similarity Heatmap
sns.heatmap(user_sim_df.iloc[:25, :25], 
            cmap="Blues",
            mask=np.eye(25, dtype=bool))  # Hide diagonal

# Distribution Plots
plt.hist(ratings['rating'], bins=20, edgecolor='black')
plt.title("Rating Distribution")

# Bar Charts: Recommendations
recommendations = recommend_movies(user_id, ...)
plt.barh(recommendations['title'], recommendations['predicted_rating'])
```

---

## 📈 Key Metrics & Concepts

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **Sparsity** | 1 - (n_ratings / (n_users × n_movies)) | % of missing ratings |
| **RMSE** | √(Σ(y_true - y_pred)² / n) | Error magnitude in rating points |
| **MAE** | Σ\|y_true - y_pred\| / n | Average absolute error |
| **Cosine Similarity** | (u·v) / (\|\|u\|\| × \|\|v\|\|) | Angle between vectors [0, 1] |
| **Pearson Correlation** | Cov(X,Y) / (σ_X × σ_Y) | Linear relationship [-1, 1] |

---

## 🎯 When to Use Which Scenario?

### User-Based CF (Scenario 1)
✅ **Best for**: Smaller datasets, new items  
✅ **Why**: Works well when users have similar tastes  
❌ **Limitations**: "Cold start" problem for new users, doesn't scale to millions of users

### Item-Based CF (Scenario 2)
✅ **Best for**: Large catalogs, stable items  
✅ **Why**: Items change slower than users, works for sparse data  
❌ **Limitations**: Requires good item metadata/ratings

---

## 💡 Usage Example

```python
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load data
ratings = pd.read_csv("ml-100k/u.data", sep="\t", 
                      names=["user_id", "movie_id", "rating", "timestamp"])
movies = pd.read_csv("ml-100k/u.item", sep="|", encoding="latin-1", 
                     names=["movie_id", "title", ...])

# Build matrix
user_item = ratings.pivot_table(index="user_id", columns="movie_id", values="rating").fillna(0)

# Compute similarity
user_sim = pd.DataFrame(cosine_similarity(user_item),
                       index=user_item.index, columns=user_item.index)

# Recommend for user_id=1
target_user = 1
similar_users = user_sim[target_user].drop(target_user).nlargest(5)
recommendations = get_similar_users_movies(target_user, similar_users, movies)
print(recommendations)
```

---

## 📁 Dataset Structure

```
ml-100k/
├── u.data          # 100K ratings (user_id, movie_id, rating, timestamp)
├── u.item          # Movie metadata (movie_id, title, release_date, genres, ...)
├── u.user          # User demographics (user_id, age, gender, occupation, ...)
├── u.genre         # Genre list
└── README          # Data dictionary
```

**Dataset Info**:
- 100,000 ratings
- 943 users
- 1,682 movies
- Ratings: 1-5 stars
- Sparsity: ~93.7% (very sparse)

---

## 🔗 Dependencies

```python
pandas          # Data manipulation
numpy           # Numerical computing
matplotlib      # Plotting
seaborn         # Statistical visualization
scikit-learn    # Machine learning (cosine_similarity, train_test_split)
```

---

## 📝 Notes & Best Practices

1. **Handle Sparse Data**: Use zero-filling for unrated movies; consider mean-centering
2. **Choose K Wisely**: Larger K = smoother predictions but less personalized
3. **Cross-Validation**: Always use train/test split to avoid overfitting to training ratings
4. **Similarity Symmetry**: Ensure similarity matrices are symmetric
5. **Bias Correction**: In user-based CF, subtract user mean rating before weighting
6. **New Users**: Item-based often handles "cold start" better

---

## 🚀 Further Improvements

- [ ] Matrix Factorization (SVD, NMF)
- [ ] Content-Based Filtering (combine with ratings)
- [ ] Hybrid Approaches (CF + Content)
- [ ] Implicit Feedback (views, clicks vs. ratings)
- [ ] Real-time Updates (batch processing)
- [ ] A/B Testing (online evaluation)

---

**Last Updated**: 2025  
**Author**: Shared Code Repository
