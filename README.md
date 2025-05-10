# 📊 YelpHybridRecommender

A hybrid recommendation system that combines Collaborative Filtering (CF) with a feature-rich XGBoost regression model to predict Yelp review ratings. Designed for scalable, large-scale processing using PySpark RDDs and optimized through bagging across multiple random seeds.

---

## 🚀 Project Overview

This project implements a **hybrid recommender system** using:

- **Collaborative Filtering** with shrinkage-adjusted **Pearson similarity**
- **Model-based regression** using **XGBoost** with 27+ engineered features
- **Spark RDDs** for scalable parallel processing
- **Ensemble prediction** via **5-seed bagging**
- **Weighted blending** of CF and model-based predictions

The final output is a predicted rating for each user-business pair in the validation set, achieving RMSE < 0.98 on Yelp data.

---

## 🧠 Methodology

- **CF Component:**  
  Predicts ratings using Pearson correlation between businesses, adjusted with a shrinkage factor to handle sparse overlaps.

- **Model-based Component (XGBoost):**  
  Trains on features extracted from `user.json` and `business.json`, including user engagement stats, compliment counts, and business attributes.

- **Hybrid Blending:**  
  Final prediction = `alpha * CF + (1 - alpha) * XGBoost`, with `alpha = 0.3` based on empirical tuning.

---

## 📂 Files Used

| File Name                | Description                                                  |
|-------------------------|--------------------------------------------------------------|
| `yelp_train.csv`         | Training data: user_id, business_id, stars                   |
| `yelp_val.csv`           | Validation/test data: user_id, business_id                   |
| `user_sample.json`       | **Sample** user metadata (original file was too large)       |
| `business_sample.json`   | **Sample** business metadata (original file was too large)   |
| `YelpHybridRecommender.py` | Main Python script implementing the full hybrid pipeline     |
| `output.csv`     | Output file containing predictions for each test pair        |

> ⚠️ Note: The full `user.json` and `business.json` files are too large to upload on GitHub.  
> Instead, **sample versions** are provided (`user_sample.json`, `business_sample.json`) to demonstrate structure and allow basic testing.

---

## 🛠 Features Engineered

From `user.json`:
- Review count (log-scaled), number of friends, useful/funny/cool votes
- Fan count, elite years, average stars
- Compliment counts (e.g., hot, funny, writer, etc.)

From `business.json`:
- Star rating, review count (log-scaled), latitude/longitude
- Price range, and 5 boolean attributes (e.g., accepts credit cards, reservations)

---

## ⚙️ Technologies Used

- Python 3.x  
- PySpark (RDD-based architecture)  
- XGBoost (`xgboost==1.x`)  
- JSON, CSV handling  
- Math, heapq, random (for similarity + baseline estimations)

---

## 📈 Performance

- ✅ **RMSE < 0.98** on validation set  
- ✅ Robust to missing metadata (uses fallback defaults)  
- ✅ Fast prediction with Spark-based parallelism  
- ✅ Generalized via bagging over 5 random seeds: `3, 11, 29, 42, 77`

---

## 📌 How to Run

```bash
# Make sure Spark and XGBoost are installed
$ spark-submit YelpHybridRecommender.py
