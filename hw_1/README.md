# Online Shoppers Purchasing Intention Prediction

## Overview

This project explores the **Online Shoppers Purchasing Intention Dataset** and builds machine learning models to predict whether a website visitor will complete a purchase during a session.

The objective is to compare a variety of classical machine learning algorithms and identify the most effective approach for detecting purchasing sessions.

---

## Dataset

**Dataset:** Online Shoppers Purchasing Intention Dataset

The dataset contains behavioral and session-level information collected from an e-commerce website.

### Target Variable

* **Revenue**

  * `True` — the visitor completed a purchase
  * `False` — no purchase was made

### Dataset Characteristics

* 12,330 observations
* No missing values
* Binary classification problem
* Significant class imbalance:

  * Purchase: ~15%
  * No Purchase: ~85%

---

## Exploratory Data Analysis (EDA)

### Key Findings

#### Data Structure

* The dataset contains 12,330 sessions.
* No missing values were found.
* The target variable (`Revenue`) is binary.

#### Target Distribution

* Only about 15% of sessions resulted in a purchase.
* The dataset is highly imbalanced.
* Most numerical features exhibit non-normal distributions with strong right skewness.

#### Purchasing Behavior

Users who completed a purchase typically:

* Viewed more product-related pages.
* Spent more time on the website.
* Had lower Bounce Rates.
* Had lower Exit Rates.
* Showed significantly higher Page Values.

#### Categorical Features

* Returning visitors converted more frequently than new visitors.
* Conversion rates varied across months.
* Seasonal purchasing patterns were observed.

---

## Modeling Approach

Because of the strong class imbalance, **ROC-AUC** and **PR-AUC** were selected as the primary evaluation metrics.

Accuracy was not used as the main metric because a high accuracy score can still be achieved while failing to correctly identify purchasing sessions.

### Models Evaluated

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)
* Decision Tree
* Random Forest
* Extra Trees
* Gradient Boosting
* AdaBoost
* CatBoost
* XGBoost
* LightGBM

---

## Initial Model Comparison

| Model               | Accuracy | Precision | Recall |    F1 | ROC-AUC | PR-AUC |
| ------------------- | -------: | --------: | -----: | ----: | ------: | -----: |
| CatBoost            |    0.871 |     0.560 |  0.775 | 0.650 |   0.929 |  0.747 |
| Gradient Boosting   |    0.899 |     0.714 |  0.581 | 0.641 |   0.927 |  0.739 |
| XGBoost             |    0.878 |     0.586 |  0.730 | 0.650 |   0.924 |  0.723 |
| LightGBM            |    0.884 |     0.600 |  0.743 | 0.664 |   0.923 |  0.719 |
| Random Forest       |    0.871 |     0.561 |  0.775 | 0.651 |   0.922 |  0.710 |
| AdaBoost            |    0.891 |     0.701 |  0.521 | 0.598 |   0.913 |  0.639 |
| Decision Tree       |    0.824 |     0.463 |  0.825 | 0.593 |   0.890 |  0.639 |
| SVM                 |    0.865 |     0.548 |  0.751 | 0.634 |   0.898 |  0.636 |
| Logistic Regression |    0.841 |     0.491 |  0.743 | 0.592 |   0.893 |  0.622 |
| Extra Trees         |    0.798 |     0.416 |  0.741 | 0.532 |   0.865 |  0.577 |
| KNN                 |    0.875 |     0.750 |  0.291 | 0.419 |   0.843 |  0.575 |

---

## Hyperparameter Optimization

The two best-performing models were selected for hyperparameter tuning using **Optuna**:

* CatBoost
* Gradient Boosting

### Tuned Gradient Boosting

| Accuracy | Precision | Recall |    F1 | ROC-AUC | PR-AUC |
| -------: | --------: | -----: | ----: | ------: | -----: |
|    0.904 |     0.791 |  0.516 | 0.624 |   0.929 |  0.741 |

### Tuned CatBoost

| Accuracy | Precision | Recall |    F1 | ROC-AUC | PR-AUC |
| -------: | --------: | -----: | ----: | ------: | -----: |
|    0.866 |     0.545 |  0.825 | 0.656 |   0.930 |  0.740 |

---

## Results

CatBoost achieved the strongest overall performance:

* ROC-AUC: **0.930**
* Recall: **82.5%**
* F1-score: **0.656**

The model successfully identified more than 80% of purchasing sessions while maintaining competitive precision.

Gradient Boosting achieved slightly higher precision and accuracy, but CatBoost provided better buyer detection and a more balanced trade-off between Precision and Recall.

Overall, boosting-based models consistently outperformed linear, distance-based, and single-tree approaches on this dataset.

---

## Project Structure

```text
project/
│
├── data/
│
├── notebooks/
│   ├── EDA.ipynb
│   └── models.ipynb
│
├── scripts/
│   ├── database.py
│   ├── models.py
│   └── tune_models.py
│
├── shopping_sessions.db
├── requirements.txt
└── README.md
```

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* CatBoost
* XGBoost
* LightGBM
* Optuna
* SQLite

---

## Conclusion

This project demonstrates the complete workflow of a machine learning classification task:

1. Data exploration and preprocessing.
2. Exploratory data analysis (EDA).
3. Training and evaluation of multiple classification models.
4. Comparison using ROC-AUC and PR-AUC metrics.
5. Hyperparameter optimization with Optuna.
6. Selection of the best-performing model.

The results show that ensemble boosting methods, particularly CatBoost, are highly effective for predicting online purchasing behavior from session-level user data.
