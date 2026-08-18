# Machine Learning Assignment 2

## Classification Models and Streamlit Deployment

### Student Details

- **Programme:** M.Tech in Data Science
- **Course:** Machine Learning
- **Assignment:** Assignment 2

---

## 1. Problem Statement

The objective of this assignment is to implement multiple classification machine learning models on a public classification dataset, evaluate their performance using multiple evaluation metrics, compare the models, and deploy the trained models using a Streamlit web application.

---

## 2. Dataset

The Breast Cancer Wisconsin (Diagnostic) dataset from the UCI Machine Learning Repository is used.

The dataset contains:

- 569 instances
- 30 predictive features
- Binary target variable: `diagnosis`

The target classes are:

- B = Benign
- M = Malignant

The `id` column is removed during preprocessing because it is an identification field and is not used as a predictive feature.

---

## 3. Machine Learning Models

The following five classification models were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Gaussian Naive Bayes
5. Random Forest Classifier

---

## 4. Evaluation Metrics

The models were evaluated using:

- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

---

## 5. Model Performance Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9298 | 0.9246 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| Random Forest | 0.9649 | 0.9942 | 1.0000 | 0.9048 | 0.9500 | 0.9258 |

---

## 6. Observations

### Logistic Regression

Logistic Regression achieved an accuracy of 0.9649 and the highest AUC of 0.9960. It also achieved a strong F1 score of 0.9512, indicating balanced precision and recall performance.

### Decision Tree

Decision Tree achieved an accuracy of 0.9298 and an AUC of 0.9246. Its performance was lower than the other evaluated models across the selected metrics.

### K-Nearest Neighbors

KNN achieved an accuracy of 0.9561 and an AUC of 0.9823. It showed strong precision and F1 score performance.

### Gaussian Naive Bayes

Naive Bayes achieved perfect precision of 1.0000 and a high AUC of 0.9934. However, its recall of 0.8333 was lower than the other high-performing models.

### Random Forest

Random Forest achieved an accuracy of 0.9649, AUC of 0.9942, perfect precision of 1.0000, and the highest MCC of 0.9258 among the evaluated models.

---

## 7. Overall Model Selection

Based on the average performance across Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC), Logistic Regression achieved the highest overall average score of 0.9567.

The model ranking based on the average evaluation score is:

| Rank | ML Model Name | Average Score |
|---|---|---:|
| 1 | Logistic Regression | 0.9567 |
| 2 | Random Forest | 0.9566 |
| 3 | KNN | 0.9436 |
| 4 | Naive Bayes | 0.9243 |
| 5 | Decision Tree | 0.9030 |

Therefore, **Logistic Regression is selected as the overall best-performing model** for this assignment.

Logistic Regression achieved the highest AUC of 0.9960 and the highest F1 Score of 0.9512 among the evaluated models. Random Forest showed very competitive performance, with the highest MCC of 0.9258 and perfect precision of 1.0000.

---

## 8. Streamlit Application

The Streamlit application provides:

- Test CSV file upload
- Classification model selection
- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- MCC
- Confusion Matrix
- Classification Report
- Prediction Summary

---

## 9. Project Structure

```text
ML Assignment 2/
│
├── ML_Assignment_2.ipynb
├── app.py
├── requirements.txt
├── breast_cancer_wisconsin.csv
├── test_data.csv
│
└── model/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    └── scaler.pkl
